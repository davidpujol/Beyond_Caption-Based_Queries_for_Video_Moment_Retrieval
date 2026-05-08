import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math

from ld_detr.transformer import T2V_TransformerEncoderLayer_no_global, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
from ld_detr.span_utils import span_cxw_to_xx


# initialize parameters with Xavier initialization
def _reset_parameters(model):
    for n, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


# Unimodal Encoder
class UnimodalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Inputs:
            x: features                                             (bs, L, dv)
        Return:
            x: features that have been mapped into the latent space (bs, L, d)
        """
        x = self.linear1(self.dropout1(self.norm1(x)))
        x = F.relu(x, inplace=True)
        x = self.linear2(self.dropout2(self.norm2(x)))
        return x


# Distill Align
class DistillAlign(nn.Module):
    def __init__(self, hidden_dim, queue_length=65536, temp=0.07, alpha=0.4):
        """
        Args:
            alpha: The distillation coefficient. Defaults to 0.4.
        """
        super().__init__()
        self.temp = nn.Parameter(torch.ones([]) * temp)
        self.alpha = alpha
        self.queue_length = queue_length

        # cosine similarity
        self.cos = nn.CosineSimilarity(dim=1)
        
        # prediction mlp
        self.h = nn.Linear(hidden_dim, hidden_dim)

        # register global features queues and normalize them
        self.register_buffer("vid_queue", torch.randn(hidden_dim,
                                                      queue_length))
        self.register_buffer("txt_queue", torch.randn(hidden_dim,
                                                      queue_length))
        self.vid_queue = F.normalize(self.vid_queue, dim=0)
        self.txt_queue = F.normalize(self.txt_queue, dim=0)

        # register the pointer
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # initialize parameters with Xavier initialization
        _reset_parameters(self.h)

    def forward(
        self,
        src_vid_cls,
        src_txt_cls,
        src_vid_cls_m,
        src_txt_cls_m,
        epoch_i,
        batch_idx,
        train_loader_length=0,
        is_training=False,
    ):
        """
        Inputs:
            src_vid_cls:    video global features                 (bs, d)
            src_txt_cls:    text global features                  (bs, d)
            src_vid_cls_m:  video momentum global features        (bs, d)
            src_txt_cls_m:  text momentum global features         (bs, d)
        Attributes:
            self.vid_queue: video momentum global features queues (d, queue_length)
            self.txt_queue: text momentum global features queues  (d, queue_length)
        Return:
            loss_align: multimodal alignment loss                 (1)
            loss_sim: multimodal similar loss                     (1)
        """

        # distillation coefficient warms up
        if epoch_i > 0 or train_loader_length == 0:
            alpha = self.alpha
        else:
            alpha = self.alpha * min(1, batch_idx / train_loader_length)

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

            # concate momentum global features and momentum global features queues
            vid_feat = torch.cat(
                [src_vid_cls_m.t(),
                 self.vid_queue.clone().detach()],
                dim=1)  # (d, bs+queue_length)
            txt_feat = torch.cat(
                [src_txt_cls_m.t(),
                 self.txt_queue.clone().detach()],
                dim=1)  # (d, bs+queue_length)

            # calculate similarity matrices
            sim_v2t_m = src_vid_cls_m @ txt_feat / self.temp  # (bs, bs+queue_length)
            sim_t2v_m = src_txt_cls_m @ vid_feat / self.temp  # (bs, bs+queue_length)

            # get an identity matrix
            sim_targets = torch.zeros(sim_v2t_m.size()).to(src_vid_cls.device)
            sim_targets.fill_diagonal_(1)  # (bs, bs+queue_length)

            # distill the similarity matrices into the identity matrices as the target matrices
            sim_v2t_targets = (alpha * F.softmax(sim_v2t_m, dim=1) +
                               (1 - alpha) * sim_targets
                               )  # (bs, bs+queue_length)
            sim_t2v_targets = (alpha * F.softmax(sim_t2v_m, dim=1) +
                               (1 - alpha) * sim_targets
                               )  # (bs, bs+queue_length)

        # calculate similarity matrices
        sim_v2t = src_vid_cls @ txt_feat / self.temp  # (bs, bs+queue_length)
        sim_t2v = src_txt_cls @ vid_feat / self.temp  # (bs, bs+queue_length)

        # cross entropy losses
        loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1) * sim_v2t_targets,
                              dim=1).mean()  # (1)
        loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1) * sim_t2v_targets,
                              dim=1).mean()  # (1)

        # multimodal alignment loss
        loss_align = (loss_v2t + loss_t2v) / 2  # (1)

        # push momentum global features into momentum global features queues
        if is_training:
            self._dequeue_and_enqueue(src_vid_cls_m, src_txt_cls_m)
        
        # # predictions
        p_vid_cls = self.h(F.relu(src_vid_cls, inplace=False))
        p_txt_cls = self.h(F.relu(src_txt_cls, inplace=False))
        
        # negative cosine similarity
        loss_sim = -(self.cos(p_vid_cls, src_txt_cls).mean() + self.cos(p_txt_cls, src_vid_cls).mean()) / 2

        return loss_align, loss_sim

    # push momentum global features into momentum global features queues
    @torch.no_grad()
    def _dequeue_and_enqueue(self, vid_feat, txt_feat):
        batch_size = vid_feat.shape[0]

        if self.queue_length % batch_size == 0:
            ptr = int(self.queue_ptr)

            self.vid_queue[:, ptr:ptr + batch_size] = vid_feat.T
            self.txt_queue[:, ptr:ptr + batch_size] = txt_feat.T
            ptr = (ptr + batch_size) % self.queue_length

            self.queue_ptr[0] = ptr


# V2T Extractor
class V2TExtractor(nn.Module):
    """read arXiv:2401.02309 for more details"""
    def __init__(self, hidden_dim=256, dropout=0.1):
        super().__init__()
        w4C = torch.empty(hidden_dim, 1)
        w4Q = torch.empty(hidden_dim, 1)
        w4mlu = torch.empty(1, 1, hidden_dim)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)
        self.cqa_linear = nn.Conv1d(
            in_channels=4 * hidden_dim,
            out_channels=hidden_dim,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=True,
        )
        weight = torch.empty(hidden_dim, 1)
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.conv1d = nn.Conv1d(
            in_channels=2 * hidden_dim,
            out_channels=hidden_dim,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=True,
        )

    def forward(self, src_vid, src_txt, src_vid_mask, src_txt_mask):
        """
        Inputs:
            src_vid:      video features                 (bs, L, d)
            src_txt:      text features                  (bs, N, d)
            src_vid_mask: video features' mask           (bs, L)
            src_txt_mask: text features' mask            (bs, N)
        Return:
            output:       text-irrelevant video features (bs, L, d)

        read arXiv:2401.02309 for more details
        """
        batch_size, c_seq_len, dim = src_vid.shape
        batch_size, q_seq_len, dim = src_txt.shape
        context = self.dropout(src_vid)
        query = self.dropout(src_txt)
        subres0 = torch.matmul(context, self.w4C).expand([-1, -1, q_seq_len])
        subres1 = (torch.matmul(query, self.w4Q).transpose(1, 2).expand(
            [-1, c_seq_len, -1]))
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        score = subres0 + subres1 + subres2
        score_ = nn.Softmax(dim=2)(self.mask_logits(score,
                                                    src_txt_mask.unsqueeze(1)))
        score_t = nn.Softmax(dim=1)(self.mask_logits(
            score, src_vid_mask.unsqueeze(2)))
        score_t = score_t.transpose(1, 2)
        c2q = torch.matmul(score_, src_txt)
        q2c = torch.matmul(torch.matmul(score_, score_t), src_vid)
        feats = torch.cat(
            [src_vid, c2q,
             torch.mul(src_vid, c2q),
             torch.mul(src_vid, q2c)],
            dim=2)
        feats = feats.transpose(1, 2)
        feats = self.cqa_linear(feats)
        feats = feats.transpose(1, 2)
        alpha = torch.tensordot(src_txt, self.weight, dims=1)
        alpha = self.mask_logits(alpha, mask=src_txt_mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_src_txt = torch.matmul(src_txt.transpose(1, 2), alphas)
        pooled_src_txt = pooled_src_txt.squeeze(2)
        _, c_seq_len, _ = feats.shape
        pooled_src_txt = pooled_src_txt.unsqueeze(1).repeat(1, c_seq_len, 1)
        output = torch.cat([feats, pooled_src_txt], dim=2)
        output = output.transpose(1, 2)
        output = self.conv1d(output)
        output = output.transpose(1, 2)
        output = F.relu(output)
        return output

    def mask_logits(self, inputs, mask, mask_value=-1e30):
        mask = mask.type(torch.float32)
        return inputs + (1.0 - mask) * mask_value


# Convolutional Block
class ConvolutionalBlock(nn.Module):
    def __init__(self, hidden_dim=256, n_blocks=5):
        super().__init__()

        # define the Block
        class TheBlock(nn.Module):
            def __init__(self, hidden_dim=256):
                super().__init__()
                self.conv1 = nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.conv2 = nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                self.bn2 = nn.BatchNorm1d(hidden_dim)

            def forward(self, x):
                return F.relu(
                    self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + x)

        blocks = [TheBlock(hidden_dim) for _ in range(n_blocks)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """
        Inputs:
            x:   (L, bs, d)
        Return:
            out: (L, bs, d)
        """
        out = x
        out = out.permute(1, 2, 0)
        for i, layer in enumerate(self.blocks):
            out = layer(out)
        out = out.permute(2, 0, 1)
        return out


# Convolutional Fuser
class ConvolutionalFuser(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.1,
        activation="prelu",
        num_v2t_encoder_layers=2,
        num_encoder1_layers=2,
        num_convolutional_blocks=5,
        num_encoder2_layers=2,
        normalize_before=False,
    ):
        super().__init__()

        # V2T Extractor, read arXiv:2401.02309 for more details
        self.v2t_extractor = V2TExtractor(hidden_dim, dropout)

        # T2V Encoder, read arXiv:2303.13874 for more details
        self.t2v_encoder = TransformerEncoder(
            T2V_TransformerEncoderLayer_no_global(
                hidden_dim,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                normalize_before,
            ),
            num_v2t_encoder_layers,
            None,
        )

        # Transformer Encoder 1
        self.transformer_encoder1 = TransformerEncoder(
            TransformerEncoderLayer(
                hidden_dim,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                normalize_before,
            ),
            num_encoder1_layers,
            None,
        )

        # Convolutional Blocks
        self.convolutional_block = ConvolutionalBlock(
            hidden_dim, num_convolutional_blocks)

        # Transformer Encoder 2
        self.transformer_encoder2 = TransformerEncoder(
            TransformerEncoderLayer(
                hidden_dim,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                normalize_before,
            ),
            num_encoder2_layers,
            None,
        )

        # initialize parameters with Xavier initialization
        _reset_parameters(self.v2t_extractor)
        _reset_parameters(self.t2v_encoder)
        _reset_parameters(self.transformer_encoder1)
        _reset_parameters(self.transformer_encoder2)

    def forward(self, src_vid, src_txt, src_vid_mask, src_txt_mask, pos_vid,
                pos_txt):
        """
        Inputs:
            src_vid:      video features                      (bs, L, d)
            src_txt:      text features                       (bs, N, d)
            src_vid_mask: video features' mask                (bs, L)
            src_txt_mask: text features' mask                 (bs, N)
            pos_vid:      video features' position embeddings (bs, L, d)
            pos_txt:      text features' position embeddings  (bs, N, d)
        Return:
            memory:       multimodal features                 (L, bs, d)
        """

        # V2T Extractor, read arXiv:2401.02309 for more details
        src_vid = self.v2t_extractor(src_vid, src_txt, src_vid_mask,
                                     src_txt_mask)  # (bs, L, d)

        # V2T Encoder, read arXiv:2303.13874 for more details
        src = torch.cat([src_vid, src_txt], dim=1)  # (bs, L+N, d)
        mask = torch.cat([src_vid_mask, src_txt_mask],
                         dim=1).bool()  # (bs, L+N)
        pos = torch.cat([pos_vid, pos_txt], dim=1)  # (bs, L, d)
        video_length = src_vid.shape[1]  # L
        src = src.permute(1, 0, 2)  # (L+N, bs, d)
        pos = pos.permute(1, 0, 2)  # (L+N, bs, d)
        src = self.t2v_encoder(src,
                               src_key_padding_mask=~mask,
                               pos=pos,
                               video_length=video_length)  # (L+N, bs, d)
        src_vid = src[:video_length]  # (L, bs, d)
        src_vid_mask = mask[:, :video_length]  # (bs, L)
        pos_vid = pos[:video_length]  # (L, bs, d)

        # Transformer Encoder 1
        memory = self.transformer_encoder1(src_vid,
                                           src_key_padding_mask=~src_vid_mask,
                                           pos=pos_vid)  # (L, bs, d)

        # Convolutional Blocks
        memory = self.convolutional_block(memory)  # (L, bs, d)

        # Transformer Encoder 2
        memory = self.transformer_encoder2(memory,
                                           src_key_padding_mask=~src_vid_mask,
                                           pos=pos_vid)  # (L, bs, d)

        return memory


# Loop Decoder
class LoopDecoder(nn.Module):
    def __init__(
        self,
        args,
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.1,
        activation="prelu",
        normalize_before=False,
        keep_query_pos=False,
        num_decoder_layers=2,
        return_intermediate_dec=True,
        query_dim=2,
        query_scale_type="cond_elewise",
        modulate_t_attn=True,
        bbox_embed_diff_each_layer=False,
        num_decoder_loops=3,
    ):
        super().__init__()
        self.num_decoder_loops = num_decoder_loops

        # Transformer Decoder
        self.transformer_decoder = TransformerDecoder(
            TransformerDecoderLayer(
                args,
                hidden_dim,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                normalize_before,
                keep_query_pos=keep_query_pos,
                rm_self_attn_decoder=args.rm_self_attn_decoder
            ),
            num_decoder_layers,
            nn.LayerNorm(hidden_dim),
            return_intermediate=return_intermediate_dec,
            d_model=hidden_dim,
            query_dim=query_dim,
            keep_query_pos=keep_query_pos,
            query_scale_type=query_scale_type,
            modulate_t_attn=modulate_t_attn,
            bbox_embed_diff_each_layer=bbox_embed_diff_each_layer,
        )

        # initialize parameters with Xavier initialization
        _reset_parameters(self.transformer_decoder)

    def forward(self, tgt, memory, src_vid_mask, pos_vid, refpoint_embed):
        """
        Inputs:
            tgt:            an zero metrix, extract multimodal features (reference_point_#, bs, d)
            memory:         multimodal features                         (L, bs, d)
            src_vid_mask:   video features' mask                        (bs, L)
            pos_vid:        video features' position embeddings         (bs, L, d)
            refpoint_embed: extract multimodal feature                  (reference_point_#, bs, 2)
        Return:
            hs:             encoded multimodal features                 (decoder_layer_#, bs, reference_point_#, d)
            reference:      encoded refpoint_embed                      (decoder_layer_#, bs, reference_point_#, 2)
        """

        pos_vid = pos_vid.permute(1, 0, 2)  # (L, bs, d)

        for _ in range(self.num_decoder_loops):
            hs, reference = self.transformer_decoder(
                tgt,
                memory,
                memory_key_padding_mask=~src_vid_mask,
                pos=pos_vid,
                refpoints_unsigmoid=refpoint_embed,
            )
            tgt = hs[-1].transpose(0, 1)

            return hs, reference


# Video Moment Retrieval Prediction Head
class VideoMomentRetrievalPredictionHead(nn.Module):
    """read arXiv:2303.13874 for more details"""
    def __init__(self, hidden_dim, span_loss_type):
        super().__init__()
        self.span_loss_type = span_loss_type

        self.class_embed = nn.Linear(hidden_dim, 2)
        self.span_embed1 = nn.Linear(hidden_dim, hidden_dim)
        self.span_embed2 = nn.Linear(hidden_dim, hidden_dim)
        self.span_embed3 = nn.Linear(hidden_dim, 2)
        
    def forward(self, hs, reference):
        outputs_class = self.class_embed(hs)
        reference_before_sigmoid = self.inverse_sigmoid(reference)
        tmp = self.span_embed3(
            F.relu(self.span_embed2(F.relu(self.span_embed1(hs)))))
        outputs_coord = tmp + reference_before_sigmoid
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        pred_logits = outputs_class[-1]
        pred_spans = outputs_coord[-1]
        pred_logits_others = outputs_class[:-1]
        pred_spans_others = outputs_coord[:-1]

        return pred_logits, pred_spans, pred_logits_others, pred_spans_others

    def inverse_sigmoid(self, x, eps=1e-3):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)


# Highlight Detection Prediction Head
class HighlightDetectionPredictionHead(nn.Module):
    """read arXiv:2401.02309 for more details"""
    def __init__(self, hidden_dim=256, clip_len=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.clip_len = clip_len
        self.gru = nn.GRU(hidden_dim,
                          hidden_dim,
                          num_layers=1,
                          bidirectional=False,
                          batch_first=True)
        self.saliency_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, pred_logits, pred_spans, memory, src_vid):
        video_length = memory.shape[1]

        prob = F.softmax(pred_logits, -1)
        scores = prob[..., 0]
        sorted_scores, sorted_indices = torch.sort(scores,
                                                   dim=-1,
                                                   descending=True)
        sorted_indices_max = sorted_indices[:, :1]

        spans = span_cxw_to_xx(pred_spans) * (video_length * self.clip_len)
        spans = torch.floor(spans / self.clip_len)

        selected_values_max = spans[torch.arange(spans.size(0)).unsqueeze(1),
                                    sorted_indices_max].squeeze(1)

        sliced_samples = []
        b = memory.size(0)
        max_time = memory.size(1)

        fixed_slice_size = max_time

        for i in range(b):
            start_time = int(selected_values_max[i, 0])
            end_time = int(selected_values_max[i, 1])
            sliced_sample = src_vid[i, start_time:end_time + 1, :]

            padding_size = fixed_slice_size - sliced_sample.size(0)
            if padding_size > 0:
                padded_slice = F.pad(sliced_sample, (0, 0, 0, padding_size),
                                     value=0)
            else:
                padded_slice = sliced_sample[:fixed_slice_size, :]

            sliced_samples.append(padded_slice)

        sliced_features = torch.stack(sliced_samples, dim=0)
        mask = sliced_features != 0.0
        _, hidden = self.gru(sliced_features)
        hidden = hidden[-1, :, :]
        sliced_features_global_expanded = hidden.unsqueeze(1)
        weight = torch.matmul(sliced_features_global_expanded,
                              src_vid.transpose(1, 2)).squeeze(1)
        memory = memory * weight.unsqueeze(-1) + memory
        saliency_scores = torch.sum(self.saliency_proj(memory),
                                    dim=-1) / np.sqrt(self.hidden_dim)

        return saliency_scores


# Prediction Heads
class PredictionHeads(nn.Module):
    def __init__(self,
                 hidden_dim,
                 span_loss_type,
                 clip_len=2,
                 aux_loss='aux_loss'):
        super().__init__()
        self.aux_loss = aux_loss

        # video moment retrieval prediction head, read arXiv:2303.13874 for more details
        self.video_moment_retrieval_prediction_head = (
            VideoMomentRetrievalPredictionHead(hidden_dim, span_loss_type))

        # highlight detection prediction head, read arXiv:2401.02309 for more details
        self.highlight_detection_prediction_head = HighlightDetectionPredictionHead(
            hidden_dim, clip_len)

    def forward(self, hs, reference, memory, src_vid):
        """
            hs:                 encoded multimodal features                    (decoder_layer_#, bs, reference_point_#, d)
            reference:          encoded refpoint_embed                         (decoder_layer_#, bs, reference_point_#, d)
            memory:             multimodal features                            (bs, L, d)
            src_vid:            video features                                 (bs, L, d)
        Return:
            pred_logits:        video moment retrieval predictions confidences (bs, reference_point_#, 2)
            pred_spans:         video moment retrieval predictions             (bs, reference_point_#, 2)
            saliency_scores:    highlight detection prediction                 (bs, L)
        """

        # video moment retrieval prediction head, read arXiv:2303.13874 for more details
        pred_logits, pred_spans, pred_logits_others, pred_spans_others = (
            self.video_moment_retrieval_prediction_head(hs, reference))

        # highlight detection prediction head, read arXiv:2401.02309 for more details
        saliency_scores = self.highlight_detection_prediction_head(
            pred_logits, pred_spans, memory, src_vid)

        # aux loss, read arXiv:2303.13874 for more details
        if self.aux_loss:
            aux_outputs = [{
                "pred_logits": a,
                "pred_spans": b
            } for a, b in zip(pred_logits_others, pred_spans_others)]

        return pred_logits, pred_spans, saliency_scores, aux_outputs
