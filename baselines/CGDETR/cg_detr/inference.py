import pprint
from tqdm import tqdm
import numpy as np
import os
from collections import defaultdict
from utils.basic_utils import AverageMeter

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from cg_detr.config import TestOptions
from cg_detr.model import build_model
from cg_detr.span_utils import span_cxw_to_xx
from cg_detr.start_end_dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from cg_detr.postprocessing_cg_detr import PostProcessorDETR
from standalone_eval.comparable_eval import eval_submission
from utils.basic_utils import save_jsonl, save_json
from utils.temporal_nms import temporal_nms

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def post_processing_mr_nms(mr_res, nms_thd, max_before_nms, max_after_nms, min_conf):
    mr_res_after_nms = []
    for e in mr_res:
        e["pred_relevant_windows"] = temporal_nms(
            e["pred_relevant_windows"][:max_before_nms],
            nms_thd=nms_thd,
            max_after_nms=max_after_nms,
            min_conf=min_conf
        )
        mr_res_after_nms.append(e)
    return mr_res_after_nms


def merge_gt_data(gt_data):
    # ------------------------------------
    # Merge GT annotations by qid and vid to support sliding window evaluation
    # This aggregates all GT relevant windows per (qid, vid) into one list
    
    merged_gt = {}  # keyed by (qid, vid)
    
    for clip_gt in gt_data:
        qid = clip_gt['qid']   # Use original_qid so that it also works when we do not want to evaluate the merged ones. After the changes in the dataset generator, by default this will be just like qid unless it has been defined as otherwise
        original_qid = clip_gt['original_qid'] #original_qid = clip_gt['original_qid']  # If the original_qid is not defined in the dataset, then this is just the same as qid
        vid = clip_gt['vid']
        cut_start = clip_gt.get('cut_start', 0)  # in seconds
        
        # Each ground truth window normalized within clip (e.g., range [0, duration])
        for window, seg_to_eval in zip(clip_gt['relevant_windows'], clip_gt['segs_to_eval']):
            # window = [start, end] normalized in [0, duration] or [0,1] scaled by duration
            abs_start = window[0] + (cut_start / 3)
            abs_end = window[1] + (cut_start / 3)
            
            if original_qid+':'+qid+':'+vid not in merged_gt:
                merged_gt[original_qid+':'+qid+':'+vid] = {}
                merged_gt[original_qid+':'+qid+':'+vid]['relevant_windows'] = []
                merged_gt[original_qid+':'+qid+':'+vid]['segs_to_eval'] = []
                merged_gt[original_qid+':'+qid+':'+vid]['num_gt_merged_window'] = 0
            
            # So we are now merging all GT windows for the same (original_qid, qid, vid) into a single list (so this is only effectively merging when original_qid = qid)
            merged_gt[original_qid+':'+qid+':'+vid]['relevant_windows'].append([abs_start, abs_end])
            merged_gt[original_qid+':'+qid+':'+vid]['segs_to_eval'].append(seg_to_eval)

        
        merged_gt[original_qid+':'+qid+':'+vid]['num_gt_merged_window'] = max(merged_gt[original_qid+':'+qid+':'+vid]['num_gt_merged_window'], clip_gt['num_gt_merged_window'])  # Just keep the same value, it is not very relevant here
    
        
    # Convert to list or dict format for eval_submission
    gt_merged_list = []
    for key, value in merged_gt.items():
        original_qid, qid, vid = key.split(':')
        gt_merged_list.append({
            'qid': qid,
            'original_qid': original_qid,
            'vid': vid,
            'relevant_windows': value['relevant_windows'],
            'segs_to_eval': value['segs_to_eval'],
            'num_gt_merged_window': value['num_gt_merged_window']
        })
        
    return gt_merged_list

# ---------------------------------------------


def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename, cuts_type):
    # IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    submission_path_no_suffix = os.path.join(opt.results_dir, save_submission_filename)

    submission_path = submission_path_no_suffix.replace(".jsonl", opt.suffix + ".jsonl")    # Add also our suffix
    save_jsonl(submission, submission_path)

    if cuts_type == 'sliding':
        # ------------------------------------
        # Added by me
        # In order to support the sliding window evaluation we need to merge the GT data for each of the videos into a single one
        gt_data = merge_gt_data(gt_data)
        # ------------------------------------
        
    if opt.eval_split_name in ["val"]:  # since test_public has no GT
        metrics = eval_submission(
            submission, gt_data,
            verbose=opt.debug, match_number=not opt.debug
        )
        save_metrics_path = submission_path_no_suffix.replace(".jsonl", "_metrics" + opt.suffix + ".json")    # Add also our suffix
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path, ]

    if opt.nms_thd != -1:
        logger.info("[MR] Performing nms with nms_thd {}".format(opt.nms_thd))
        submission_after_nms = post_processing_mr_nms(
            submission, nms_thd=opt.nms_thd,
            max_before_nms=opt.max_before_nms, max_after_nms=opt.max_after_nms, min_conf=opt.conf_thd,
        )

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path_no_suffix.replace(".jsonl", "_nms_thd_" + str(opt.nms_thd) + opt.suffix + ".jsonl")
        save_jsonl(submission_after_nms, submission_nms_path)
        if opt.eval_split_name == "val":
            metrics_nms = eval_submission(
                submission_after_nms, gt_data,
                verbose=opt.debug, match_number=not opt.debug
            )
            save_metrics_nms_path = submission_path_no_suffix.replace(".jsonl", "_nms_thd_" + str(opt.nms_thd) + "_metrics" + opt.suffix + ".jsonl")

            save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False)
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [submission_nms_path, ]
    else:
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths


# for HL
@torch.no_grad()
def compute_hl_results(model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None):
    model.eval()
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    topk = 5 # top-5 map

    video_ap_collected = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]

        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

        outputs = model(**model_inputs)

        preds = outputs['saliency_scores'].clone().detach()

        for meta, pred in zip(query_meta, preds):
            pred = pred
            label = meta['label'] # raw label

            video_ap = []
            # Follow the UMT code "https://github.com/TencentARC/UMT/blob/main/datasets/tvsum.py"
            
            if opt.dset_name in ["tvsum"]:
                for i in range(20):
                    pred=pred.cpu()
                    cur_pred = pred[:len(label)]
                    inds = torch.argsort(cur_pred, descending=True, dim=-1)

                    # video_id = self.get_video_id(idx)
                    cur_label = torch.Tensor(label)[:, i]
                    cur_label = torch.where(cur_label > cur_label.median(), 1.0, .0)

                    cur_label = cur_label[inds].tolist()[:topk]

                    # if (num_gt := sum(cur_label)) == 0:
                    num_gt = sum(cur_label)
                    if num_gt == 0:
                        video_ap.append(0)
                        continue

                    hits = ap = rec = 0
                    prc = 1

                    for j, gt in enumerate(cur_label):
                        hits += gt

                        _rec = hits / num_gt
                        _prc = hits / (j + 1)

                        ap += (_rec - rec) * (prc + _prc) / 2
                        rec, prc = _rec, _prc

                    video_ap.append(ap)
            
            elif opt.dset_name in ["youtube_uni"]:
                cur_pred = pred[:len(label)]
                # if opt.dset_name == "tvsum_sfc":
                cur_pred = cur_pred.cpu()
                inds = torch.argsort(cur_pred, descending=True, dim=-1)


                cur_label = torch.Tensor(label).squeeze()[inds].tolist()
                
                num_gt = sum(cur_label)
                if num_gt == 0:
                    video_ap.append(0)
                    continue

                hits = ap = rec = 0
                prc = 1

                for j, gt in enumerate(cur_label):
                    hits += gt

                    _rec = hits / num_gt
                    _prc = hits / (j + 1)

                    ap += (_rec - rec) * (prc + _prc) / 2
                    rec, prc = _rec, _prc
                
                video_ap.append(float(ap))
            else:
                print("No such dataset")
                exit(-1)
                    
            video_ap_collected.append(video_ap)

    mean_ap = np.mean(video_ap_collected)
    submmission = dict(mAP=round(mean_ap, 5))
    

    # tensorboard writer
    if write_tb and criterion:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    return submmission, loss_meters 



@torch.no_grad()
def compute_mr_results(model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None):
    model.eval()
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]

        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

        outputs = model(**model_inputs)
        prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #queries, #classes=2)
        if opt.span_loss_type == "l1":
            scores = prob[..., 0]  # * (batch_size, #queries)  foreground label is 0, we directly take it
            
            # Combine the scores with actionness score
            if 'pred_actionness' in outputs:
                actionness_score = outputs['pred_actionness']  # b, q, 1
                #scores = actionness_score[:,:,0]
                scores = (scores * actionness_score[:,:,0]).sqrt()  # (bsz, #queries)
            
            pred_spans = outputs["pred_spans"]  # (bsz, #queries, 2)
            _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
            saliency_scores = []
            valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
            for j in range(len(valid_vid_lengths)):
                saliency_scores.append(_saliency_scores[j, :int(valid_vid_lengths[j])].tolist())
        else:
            bsz, n_queries = outputs["pred_spans"].shape[:2]  # # (bsz, #queries, max_v_l *2)
            pred_spans_logits = outputs["pred_spans"].view(bsz, n_queries, 2, opt.max_v_l)
            pred_span_scores, pred_spans = F.softmax(pred_spans_logits, dim=-1).max(-1)  # 2 * (bsz, #queries, 2)
            scores = torch.prod(pred_span_scores, 2)  # (bsz, #queries)
            pred_spans[:, 1] += 1
            pred_spans *= opt.clip_length

        # compose predictions
        for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans.cpu(), scores.cpu())):
            if opt.span_loss_type == "l1":
                spans = span_cxw_to_xx(spans) * meta["duration"]
                spans = torch.clamp(spans, 0, meta["duration"])
            # # (#queries, 3), [st(float), ed(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            if not opt.no_sort_results:
                cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                qid=meta["qid"], # Again, use this original_qid so that it works both for merged and non-merged evaluations
                original_qid=meta.get("original_qid", meta["qid"]),  # If the original_qid is not defined in the dataset, then this is just the same as qid
                query=meta["query"],
                vid=meta["vid"],
                pred_relevant_windows=cur_ranked_preds,
                pred_saliency_scores=saliency_scores[idx]
            )
            mr_res.append(cur_query_pred)

        if criterion:
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_dict["loss_overall"] = float(losses)  # for logging only
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        if opt.debug:
            break

    if write_tb and criterion:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    if opt.dset_name in ['hl']:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length, min_ts_val=0, max_ts_val=150,
            min_w_l=2, max_w_l=150, move_window_method="left",
            process_func_names=("clip_ts", "round_multiple")
        )
    elif opt.dset_name in ['charadesSTA']:
        if opt.v_feat_dim == 4096: # vgg
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length, min_ts_val=0, max_ts_val=360,
                min_w_l=12, max_w_l=360, move_window_method="left",
                process_func_names=("clip_ts", "round_multiple")
            )
        else:
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length, min_ts_val=0, max_ts_val=150,
                min_w_l=2, max_w_l=60, move_window_method="left",
                process_func_names=("clip_ts", "round_multiple")
            )
    else:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length, min_ts_val=0, max_ts_val=50000,
            min_w_l=0, max_w_l=50000, move_window_method="left",
            process_func_names=(["round_multiple"])
        )

    mr_res = post_processor(mr_res)
    return mr_res, loss_meters


# -------------------------------------------------
# Implemented by me to do multi-cut evaluation.
# This applies each of the cuts independently and then merges the results which are used for the evaluation

@torch.no_grad()
def compute_mr_results_multi_moment(model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None):
    model.eval()
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    # Store per (qid, vid) all clip predictions before aggregation
    # Key: (qid, vid), Value: list of predictions [[start, end, score], ...] from all overlapping clips
    all_predictions = defaultdict(list)
    # Store other info per qid-vid (e.g., query text, saliency score from clips)
    all_query_meta = {}

    for batch in tqdm(eval_loader, desc="compute st ed scores with sliding windows"):
        query_meta = batch[0]

        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

        outputs = model(**model_inputs)
        prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #queries, #classes=2)

        if opt.span_loss_type == "l1":
            scores = prob[..., 0]  # * (batch_size, #queries)  foreground label is 0, we directly take it (original)
            
            # Combine the scores with actionness score
            if 'pred_actionness' in outputs:
                actionness_score = outputs['pred_actionness']  # b, q, 1
                #scores = actionness_score[:,:,0]
                scores = (scores * actionness_score[:,:,0]).sqrt()  # (bsz, #queries)
            
            pred_spans = outputs["pred_spans"]  # (bsz, #queries, 2)
            
            _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
            saliency_scores = []
            valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
            for j in range(len(valid_vid_lengths)):
                saliency_scores.append(_saliency_scores[j, :int(valid_vid_lengths[j])].tolist())
        else:
            bsz, n_queries = outputs["pred_spans"].shape[:2]  # # (bsz, #queries, max_v_l *2)
            pred_spans_logits = outputs["pred_spans"].view(bsz, n_queries, 2, opt.max_v_l)
            pred_span_scores, pred_spans = F.softmax(pred_spans_logits, dim=-1).max(-1)  # 2 * (bsz, #queries, 2)
            scores = torch.prod(pred_span_scores, 2)  # (bsz, #queries)
            pred_spans[:, 1] += 1
            pred_spans *= opt.clip_length

        
        # For each sample in batch (each clip), adjust predicted spans by clip start time to map to full video
        for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans.cpu(), scores.cpu())):
            clip_start = meta["cut_start"]  # You must provide clip start time in metadata when preparing loader
            duration = meta["duration"]
            qid = meta["qid"]
            original_qid = meta["original_qid"]
            vid = meta["vid"]

            if opt.span_loss_type == "l1":
                spans = span_cxw_to_xx(spans) * meta["duration"] # Convert to relatie time in seconds
                spans = torch.clamp(spans, 0, duration) # shape (). It's a single scalar
                
                # Convert to absolute time by adding clip start
                spans = spans + (clip_start / 3)    # Add the clip start time (in seconds) to each span
            cur_ranked_preds = torch.cat([spans, score[:,None]], dim=-1).tolist()
    
            if not opt.no_sort_results:
                cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]

            # Optional: Add the predictions only if there is at least one valid GT segment in this window
            if meta["relevant_windows"].shape[0] == 0:
                continue
                
            # Aggregate per (qid, vid) -> This allows to consider jointly all the predictions that share the same original_qid (which means that are just cuts of the same video that correspond to a given qid)
            all_predictions[(original_qid, vid)].extend(cur_ranked_preds)
            
            # Save meta once (queries and saliency can be updated or just keep last)
            all_query_meta[(original_qid, vid)] = dict(
                qid=qid,
                original_qid=original_qid,
                query=meta["query"],
                vid=vid,
                pred_saliency_scores=saliency_scores[idx] if opt.span_loss_type == "l1" else None
            )
        
        if opt.debug:
            break

    if write_tb and criterion:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    # Compose final aggregated predictions list for post-processing
    mr_res = []
    for (original_qid, vid), preds in all_predictions.items():
        # Optionally sort once more here by score descending ()
        #preds = sorted(preds, key=lambda x: x[2], reverse=True)
        
        query_meta = all_query_meta[(original_qid, vid)]
        mr_res.append(dict(
            qid=query_meta["qid"],
            original_qid=original_qid,
            query=query_meta["query"],
            vid=vid,
            pred_relevant_windows=preds,
            pred_saliency_scores=query_meta.get("pred_saliency_scores")
        ))

    # Setup post-processor consistently as before
    if opt.dset_name in ['hl']:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length, min_ts_val=0, max_ts_val=150,
            min_w_l=2, max_w_l=150, move_window_method="left",
            process_func_names=("clip_ts", "round_multiple")
        )
    elif opt.dset_name in ['charadesSTA']:
        if opt.v_feat_dim == 4096:
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length, min_ts_val=0, max_ts_val=360,
                min_w_l=12, max_w_l=360, move_window_method="left",
                process_func_names=("clip_ts", "round_multiple")
            )
        else:
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length, min_ts_val=0, max_ts_val=150,
                min_w_l=2, max_w_l=60, move_window_method="left",
                process_func_names=("clip_ts", "round_multiple")
            )
    else:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length, min_ts_val=0, max_ts_val=50000,
            min_w_l=0, max_w_l=50000, move_window_method="left",
            process_func_names=(["round_multiple"])
        )

    # Post-process the aggregated predictions from overlapping clips
    mr_res = post_processor(mr_res)

    return mr_res, loss_meters

# -------------------------------------------------


def get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer):
    """compute and save query and video proposal embeddings"""
    eval_res, eval_loss_meters = compute_mr_results(model, eval_loader, opt, epoch_i, criterion, tb_writer)  # list(dict)
    return eval_res, eval_loss_meters


def eval_epoch(model, eval_dataset, opt, save_submission_filename, epoch_i=None, criterion=None, tb_writer=None):
    logger.info("Generate submissions")
    model.eval()
    if criterion is not None and eval_dataset.load_labels:
        criterion.eval()
    else:
        criterion = None

    if opt.dset_name == 'tacos':
        shuffle = True
    else:
        shuffle = False

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=shuffle,
        pin_memory=opt.pin_memory
    )

    if eval_dataset.cuts_type in ['single', 'multi']:
        # Standard evaluation where we evaluate each of the clip entries independently
        submission, eval_loss_meters = get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer)
    elif eval_dataset.cuts_type == 'sliding':
        # Multi-cut evaluation where we evaluate each of the sliding windows independently
        submission, eval_loss_meters = compute_mr_results_multi_moment(model, eval_loader, opt, epoch_i, criterion, tb_writer)
    else:
        raise ValueError(f"Unknown cuts_type {eval_dataset.cuts_type} for evaluation")
    
    
    if opt.no_sort_results:
        save_submission_filename = save_submission_filename.replace(".jsonl", "_unsorted.jsonl")
    metrics, metrics_nms, latest_file_paths = eval_epoch_post_processing(
        submission, opt, eval_dataset.data, save_submission_filename, cuts_type=eval_dataset.cuts_type)
    return metrics, metrics_nms, eval_loss_meters, latest_file_paths


def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/optimizer/scheduler")
    model, criterion = build_model(opt)
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)

    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)

    if opt.resume is not None:
        logger.info(f"Load checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu",weights_only=False)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        model.load_state_dict(checkpoint["model"])
        if opt.resume_all:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            opt.start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Loaded model saved at epoch {checkpoint['epoch']} from checkpoint: {opt.resume}")
    else:
        logger.warning("If you intend to evaluate the model, please specify --resume with ckpt path")

    return model, criterion, optimizer, lr_scheduler


def start_inference():
    
    opt = TestOptions().parse()
    split = opt.split
    if split is not None:
        opt.eval_split_name = split

    logger.info("Setup config, data and model...")

    cudnn.benchmark = True
    cudnn.deterministic = False

    assert opt.eval_path is not None
    if opt.eval_split_name == 'val':
        loadlabel = True
    else:
        loadlabel = False

    eval_dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=loadlabel,  # opt.eval_split_name == "val",
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        dset_domain=opt.dset_domain,
        cuts_type='sliding' if opt.dset_name in ['hd_epic', 'yc2'] else 'single',   # Options are sliding, single, multi
        is_training=False,
    )

    model, criterion, _, _ = setup_model(opt)

    save_submission_filename = "hl_{}_submission.jsonl".format(
        opt.eval_split_name)
    
    logger.info("Starting inference...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
            eval_epoch(model, eval_dataset, opt, save_submission_filename, criterion=criterion)
    if opt.eval_split_name == 'val':
        logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
    if metrics_nms is not None:
        logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

def main():
    # Call your start_inference with the right arguments
    start_inference()

if __name__ == '__main__':
    main()