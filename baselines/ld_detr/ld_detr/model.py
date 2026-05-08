import torch
import torch.nn.functional as F
from torch import nn

from ld_detr.span_utils import generalized_temporal_iou, span_cxw_to_xx
from ld_detr.matcher import build_matcher
from ld_detr.position_encoding import build_position_encoding
from ld_detr.misc import accuracy

from ld_detr.methods import UnimodalEncoder, DistillAlign, ConvolutionalFuser, LoopDecoder, PredictionHeads


class LD_DETR(nn.Module):
    """LD-DETR."""
    def __init__(
        self,
        args,
        txt_dim,
        vid_dim,
        hidden_dim,
        num_queries,
        aux_loss=False,
        position_embedding="sine",
        max_v_l=75,
        max_q_l=32,
        span_loss_type="l1",
        use_txt_pos=False,
        aud_dim=0,
        queue_length=65536,
        momentum=0.995,
        distillation_coefficient=0.4,
        num_v2t_encoder_layers=2,
        num_encoder1_layers=2,
        num_convolutional_blocks=5,
        num_encoder2_layers=2,
        num_decoder_layers=2,
        num_decoder_loops=3,
        clip_len=2,
    ):
        super().__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.max_q_l = max_q_l
        self.use_txt_pos = use_txt_pos
        self.momentum = momentum
        self.clip_len = clip_len

        # positional embedding
        self.position_embed, self.txt_position_embed = build_position_encoding(
            hidden_dim, position_embedding, max_q_l)

        # unimodal encoders
        self.video_encoder = UnimodalEncoder(vid_dim + aud_dim, hidden_dim)
        self.text_encoder = UnimodalEncoder(txt_dim, hidden_dim)
        self.momentum_video_encoder = UnimodalEncoder(vid_dim + aud_dim,
                                                      hidden_dim)
        self.momentum_text_encoder = UnimodalEncoder(txt_dim, hidden_dim)
        self.model_pairs = [
            [self.video_encoder, self.momentum_video_encoder],
            [self.text_encoder, self.momentum_text_encoder],
        ]
        self._copy_params()

        # distill align
        self.distill_align = DistillAlign(hidden_dim,
                                          queue_length=queue_length,
                                          alpha=distillation_coefficient)

        # convolutional fuser
        self.convolutional_fuser = ConvolutionalFuser(
            hidden_dim=hidden_dim,
            num_v2t_encoder_layers=num_v2t_encoder_layers,
            num_encoder1_layers=num_encoder1_layers,
            num_convolutional_blocks=num_convolutional_blocks,
            num_encoder2_layers=num_encoder2_layers,
        )

        # loop decoder
        self.query_embed = nn.Embedding(num_queries, 2)
        self.loop_decoder = LoopDecoder(
            args=args,
            hidden_dim=hidden_dim,
            num_decoder_layers=num_decoder_layers,
            num_decoder_loops=num_decoder_loops,
        )

        # prediction heads
        self.prediction_heads = PredictionHeads(hidden_dim, span_loss_type,
                                                clip_len, aux_loss)

    # ------------------------------
    # Also added by me to optionally mask out some of the queries
    def dropout_queries(self, q, drop_rate=0.0):
        """
        Drop exactly (drop_rate * Q) queries.
        q: Tensor [Q, F] (learned query embeddings)
        
        Returns:
            kept_q: [Q_kept, F] subset of queries
            keep_mask: [Q] boolean mask, True = kept
        """
        Q = q.shape[0]
        if drop_rate <= 0.0 or not self.training:
            keep_mask = torch.ones(Q, dtype=torch.bool, device=q.device)
            return q, keep_mask

        # Number of queries to drop
        num_drop = int(round(drop_rate * Q))
        num_keep = Q - num_drop

        # Randomly choose which queries to keep
        perm = torch.randperm(Q, device=q.device)
        keep_idx = perm[:num_keep]
        keep_mask = torch.zeros(Q, dtype=torch.bool, device=q.device)
        keep_mask[keep_idx] = True
        kept_q = q[keep_mask]
        return kept_q, keep_mask

    def forward(
        self,
        src_txt,
        src_txt_mask,
        src_vid,
        src_vid_mask,
        vid=None,
        qid=None,
        src_aud=None,
        src_aud_mask=None,
        epoch_i=0,
        batch_idx=0,
        train_loader_length=0,
        targets=None,
        is_training=False,
    ):
        """
        Inputs:
            src_vid:      video features                                                      (bs, L, d_l)
            src_txt:      text features                                                       (bs, N, d_n)
            src_vid_mask: video features' mask, when == False means this part is not in video (bs, L)
            src_txt_mask: text features' mask                                                 (bs, N)
            src_aud:      audio features                                                      (bs, L, d_a)
            src_aud_mask: audio features' mask                                                (bs, L)
        """

        # ----------------
        # Before anything, mask out some of the learnable queries
        # Note that this removes it altogether. So this would go from shape (n_queries,2) to (n_queries*drop_rate, 2)
        query_embed, _ = self.dropout_queries(self.query_embed.weight, drop_rate=self.args.qd_rate)
        # ----------------
        


        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)

        # unimodal encoders
        src_vid_copy = src_vid  # (bs, L, d_l+da)
        src_txt_copy = src_txt  # (bs, N, d_n)
        src_vid = self.video_encoder(src_vid)  # (bs, L, d)
        src_txt = self.text_encoder(src_txt)  # (bs, N, d)
        with torch.no_grad():
            self._momentum_update()
            src_vid_m = self.momentum_video_encoder(src_vid_copy)  # (bs, L, d)
            src_txt_m = self.momentum_text_encoder(src_txt_copy)  # (bs, N, d)

        # distill align
        loss_align, loss_sim = self.distill_align(
            F.normalize(src_vid.mean(1), dim=-1),
            F.normalize(src_txt.mean(1), dim=-1),
            F.normalize(src_vid_m.mean(1), dim=-1).detach(),
            F.normalize(src_txt_m.mean(1), dim=-1).detach(),
            epoch_i=epoch_i,
            batch_idx=batch_idx,
            train_loader_length=train_loader_length,
            is_training=is_training,
        )  # (1), (1)

        # positional embedding
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bs, L, d)
        pos_txt = (self.txt_position_embed(src_txt) if self.use_txt_pos else
                   torch.zeros_like(src_txt))  # (bs, N, d)
        
        # convolutional fuser
        memory = self.convolutional_fuser(src_vid, src_txt,
                                          src_vid_mask.bool(),
                                          src_txt_mask.bool(), pos_vid,
                                          pos_txt)  # (L, bs, d)

        # loop decoder
        _, bs, d = memory.shape
        refpoint_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # (reference_point_#, bs, 2)
        tgt = torch.zeros(refpoint_embed.shape[0], bs,
                          d).cuda()  # (reference_point_#, bs, d)
        hs, reference = self.loop_decoder(
            tgt, memory, src_vid_mask.bool(), pos_vid, refpoint_embed
        )  # (decoder_layer_#, bs, reference_point_#, d), (decoder_layer_#, bs, reference_point_#, 2)
        memory = memory.transpose(0, 1)  # (bs, L, d)

        # prediction heads
        pred_logits, pred_spans, saliency_scores, aux_outputs = self.prediction_heads(
            hs, reference, memory, src_vid)
        
        # losses
        out = {}
        out["loss_align"] = loss_align
        out["loss_sim"] = loss_sim
        out["video_mask"] = src_vid_mask
        out["pred_logits"] = pred_logits  # (bs, reference_point_#, 2)
        out["pred_spans"] = pred_spans  # (bs, reference_point_#, 2)
        out["saliency_scores"] = saliency_scores  # (bs, L)
        out["aux_outputs"] = aux_outputs

        return out

    @torch.no_grad()
    def _copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(),
                                      model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(),
                                      model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1.0 - self.momentum)


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(
        self,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        span_loss_type,
        max_v_l,
        saliency_margin=1,
        use_matcher=True,
    ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = (
            self.eos_coef
        )  # lower weight for background (index 1, foreground index 0)
        self.register_buffer("empty_weight", empty_weight)

        # for tvsum,
        self.use_matcher = use_matcher

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
        The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert "pred_spans" in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs["pred_spans"][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat(
            [t["spans"][i] for t, (_, i) in zip(targets, indices)],
            dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction="none")
            loss_giou = 1 - torch.diag(
                generalized_temporal_iou(span_cxw_to_xx(src_spans),
                                         span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2,
                                       self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction="none")
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses["loss_span"] = loss_span.mean()
        losses["loss_giou"] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]  # (batch_size, #queries, #classes=2)
        
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.background_label,
            dtype=torch.int64,
            device=src_logits.device,
        )  # (batch_size, #queries) # 1
        
        target_classes[idx] = self.foreground_label  # 0
        
        # Class weights for CE: [foreground=1, background=eos_coef]
        class_weights = torch.tensor([1.0, 0.1], device=src_logits.device)
        loss = F.cross_entropy(
            src_logits.transpose(1, 2),  # (B, C, Q)
            target_classes,              # (B, Q)
            weight=class_weights,
            reduction="none"
            )
        
        losses = {"loss_label": loss.mean()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = (
                100 - accuracy(src_logits[idx], self.foreground_label)[0])
            
            try:
                # Compute the accuracy of the background class
                background_idx = (target_classes == self.background_label).nonzero(as_tuple=True)
                background_accuracy = 100 - accuracy(src_logits[background_idx], self.background_label)[0]
                losses['background_error'] = background_accuracy.item()
            except:
                pass
            
        return losses
 

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        vid_token_mask = outputs["video_mask"]
        saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
        saliency_contrast_label = targets["saliency_all_labels"]
        saliency_scores = (vid_token_mask * saliency_scores +
                           (1.0 - vid_token_mask) * -1e3)

        tau = 0.5
        loss_rank_contrastive = 0.0

        # for rand_idx in range(1, 13, 3):
        #     # 1, 4, 7, 10 --> 5 stages
        for rand_idx in range(1, 12):
            drop_mask = ~(saliency_contrast_label > 100)  # no drop
            pos_mask = (saliency_contrast_label >= rand_idx
                        )  # positive when equal or higher than rand_idx

            if torch.sum(pos_mask) == 0:  # no positive sample
                continue
            else:
                batch_drop_mask = (torch.sum(pos_mask, dim=1) > 0
                                   )  # negative sample indicator

            # drop higher ranks
            cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e3

            # numerical stability
            logits = (cur_saliency_scores -
                      torch.max(cur_saliency_scores, dim=1, keepdim=True)[0])

            # softmax
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(
                exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask
                                 ).sum(1) / (pos_mask.sum(1) + 1e-6)

            loss = -mean_log_prob_pos * batch_drop_mask

            loss_rank_contrastive = loss_rank_contrastive + loss.mean()

        loss_rank_contrastive = loss_rank_contrastive / 12

        # margin: Moment-DETR
        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(
            saliency_scores.device)
        pos_scores = torch.stack(
            [
                saliency_scores[batch_indices, pos_indices[:, col_idx]]
                for col_idx in range(num_pairs)
            ],
            dim=1,
        )
        neg_scores = torch.stack(
            [
                saliency_scores[batch_indices, neg_indices[:, col_idx]]
                for col_idx in range(num_pairs)
            ],
            dim=1,
        )
        loss_saliency = (torch.clamp(
            self.saliency_margin + neg_scores - pos_scores, min=0).sum() /
                         (len(pos_scores) * num_pairs) * 2
                         )  # * 2 to keep the loss the same scale
        loss_saliency = loss_saliency + loss_rank_contrastive
        return {"loss_saliency": loss_saliency}

    def loss_align(self, outputs, targets, indices, log=True):
        """multimodal alignment loss"""
        loss = outputs["loss_align"]
        losses = {"loss_align": loss}
        return losses

    def loss_sim(self, outputs, targets, indices, log=True):
        """multimodal similar loss"""
        loss = outputs["loss_sim"]
        losses = {"loss_sim": loss}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "align": self.loss_align,
            "sim": self.loss_sim,
            "saliency": self.loss_saliency,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items() if k != "aux_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)

        # only for HL, do not use matcher
        if self.use_matcher:
            indices = self.matcher(outputs_without_aux, targets)
            losses_target = self.losses
        else:
            indices = None
            losses_target = ["saliency"]

        # Compute all the requested losses
        losses = {}
        # for loss in self.losses:
        for loss in losses_target:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # indices = self.matcher(aux_outputs, targets)
                if self.use_matcher:
                    indices = self.matcher(aux_outputs, targets)
                    losses_target = self.losses
                else:
                    indices = None
                    losses_target = ["saliency"]
                # for loss in self.losses:
                for loss in losses_target:
                    if loss in [
                            "align",
                            "sim",
                            "saliency",
                    ]:  # skip as it is only in the top layer
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices,
                                           **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


def build_model(args):
    device = torch.device(args.device)

    if args.a_feat_dir is None:
        model = LD_DETR(
            args = args,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            hidden_dim=args.hidden_dim,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            queue_length=args.queue_length,
            distillation_coefficient=args.distillation_coefficient,
            num_v2t_encoder_layers=args.num_v2t_encoder_layers,
            num_encoder1_layers=args.num_encoder1_layers,
            num_convolutional_blocks=args.num_convolutional_blocks,
            num_encoder2_layers=args.num_encoder2_layers,
            num_decoder_layers=args.num_decoder_layers,
            num_decoder_loops=args.num_decoder_loops,
            clip_len=args.clip_length,
        )
    else:
        model = LD_DETR(
            args = args,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            aud_dim=args.a_feat_dim,
            hidden_dim=args.hidden_dim,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            queue_length=args.queue_length,
            distillation_coefficient=args.distillation_coefficient,
            num_v2t_encoder_layers=args.num_v2t_encoder_layers,
            num_encoder1_layers=args.num_encoder1_layers,
            num_convolutional_blocks=args.num_convolutional_blocks,
            num_encoder2_layers=args.num_encoder2_layers,
            num_decoder_layers=args.num_decoder_layers,
            num_decoder_loops=args.num_decoder_loops,
            clip_len=args.clip_length,
        )

    matcher = build_matcher(args)
    weight_dict = {
        "loss_span": args.span_loss_coef,
        "loss_giou": args.giou_loss_coef,
        "loss_label": args.label_loss_coef,
        "loss_align": args.align_loss_coef,
        "loss_sim": args.sim_loss_coef,
        "loss_saliency": args.lw_saliency,
    }
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.num_decoder_layers - 1):
            aux_weight_dict.update({
                k + f"_{i}": v
                for k, v in weight_dict.items() if k != "loss_saliency"
            })
        weight_dict.update(aux_weight_dict)

    losses = ["spans", "labels", "align", "sim", "saliency"]

    # For tvsum dataset
    use_matcher = not (args.dset_name == "tvsum")

    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        eos_coef=args.eos_coef,
        span_loss_type=args.span_loss_type,
        max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin,
        use_matcher=use_matcher,
    )
    criterion.to(device)
    return model, criterion
