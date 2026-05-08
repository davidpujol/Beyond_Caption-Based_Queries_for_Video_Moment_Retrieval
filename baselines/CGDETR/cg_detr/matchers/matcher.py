# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from cg_detr.span_utils import generalized_temporal_iou, span_cxw_to_xx


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self,  cost_class: float = 1, cost_span: float = 1, cost_giou: float = 1,
                 span_loss_type: str = "l1", max_v_l: int = 75):
        """Creates the matcher

        Params:
            cost_span: This is the relative weight of the L1 error of the span coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the spans in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.foreground_label = 0
        assert cost_class != 0 or cost_span != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Params:
            outputs: dict with keys
                - "pred_spans": Tensor [batch_size, num_queries, 2]
                - "pred_logits": Tensor [batch_size, num_queries, num_classes]
            targets: dict with key "span_labels", list of dicts, each dict with "spans": Tensor [num_gt_spans, 2]
        
        Returns:
            List of size batch_size with tuples (index_i, index_j) of matched prediction and gt indices.
        """
        bs, num_queries = outputs["pred_spans"].shape[:2]
        targets = targets["span_labels"]  # list of length bs, each is dict with 'spans'

        indices = []

        for b_i in range(bs):
            pred_spans = outputs["pred_spans"][b_i]  # [num_queries, 2]
            pred_logits = outputs["pred_logits"][b_i]  # [num_queries, num_classes]
            tgt = targets[b_i]
            tgt_spans = tgt["spans"]  # [num_gt_spans, 2]

            num_tgt = tgt_spans.shape[0]

            if num_tgt == 0:
                # No GT spans in this sample: no matching, all predictions are background
                indices.append((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)))
                continue

            # Compute classification cost
            out_prob = pred_logits.softmax(-1)  # [num_queries, num_classes]

            # foreground_label assumed 0, background_label 1
            tgt_ids = torch.full((num_tgt,), self.foreground_label, dtype=torch.int64, device=out_prob.device)

            # Classification cost: negative prob of gt class for all queries and targets
            cost_class = -out_prob[:, tgt_ids]  # [num_queries, num_tgt]

            # Compute span cost
            if self.span_loss_type == "l1":
                # L1 cost (cdist)
                cost_span = torch.cdist(pred_spans, tgt_spans, p=1)

                # giou cost
                cost_giou = -generalized_temporal_iou(span_cxw_to_xx(pred_spans), span_cxw_to_xx(tgt_spans))
            else:
                # For ce span loss type (if used), compute differently
                # (You can add your logic here if needed)
                cost_span = torch.zeros_like(cost_class)
                cost_giou = torch.zeros_like(cost_class)

            # Final cost matrix
            C = self.cost_class * cost_class + self.cost_span * cost_span + self.cost_giou * cost_giou
            C = C.cpu()

            # Hungarian matching for this element
            row_ind, col_ind = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64)))

        return indices


def build_matcher(args):
    return HungarianMatcher(
        cost_span=args.set_cost_span, cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_class, span_loss_type=args.span_loss_type, max_v_l=args.max_v_l
    )
