import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
import logging
from os.path import join, exists
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from ld_detr.span_utils import span_xx_to_cxw
import torch.nn as nn
import time
import os
import pickle
import torch.nn.functional as F

logger = logging.getLogger(__name__)

TVSUM_SPLITS = {
    'BK': {
        'train': ['WxtbjNsCQ8A', 'EE-bNr36nyA', 'oDXZc0tZe04', 'uGu_10sucQo'],
        'val': ['Se3oxnaPsz0']
    },
    'BT': {
        'train': ['eQu1rNs0an0', 'qqR6AEXwxoQ', 'EYqVtI9YWJA', 'iVt07TCkFM0'],
        'val': ['JgHubY5Vw3Y']
    },
    'DS': {
        'train': ['kLxoNp-UchI', 'NyBmCxDoHJU', 'jcoYJXDG9sw', '-esJrBWj2d8'],
        'val': ['E11zDS9XGzg']
    },
    'FM': {
        'train': ['_xMr-HKMfVA', 'byxOvuiIJV0', 'VuWGsYPqAX8', 'xmEERLqJ2kU'],
        'val': ['JKpqYvAdIsw']
    },
    'GA': {
        'train': ['xxdtq8mxegs', 'i3wAGJaaktw', '0tmA_C6XwfM', '3eYKfiOEJNs'],
        'val': ['Bhxk-O1Y7Ho']
    },
    'MS': {
        'train': ['Hl-__g2gn_A', 'WG0MBPpPC6I', 'LRw_obCPUt0', '37rzWOQsNIw'],
        'val': ['Yi4Ij2NM7U4']
    },
    'PK': {
        'train': ['GsAD1KT1xo8', 'XkqCExn6_Us', 'b626MiF1ew4', 'PJrm840pAUI'],
        'val': ['cjibtmSLxQ4']
    },
    'PR': {
        'train': ['RBCABdttQmI', 'z_6gVvQb2d0', '4wU_LUjG5Ic', '91IHQYk1IQM'],
        'val': ['fWutDQy1nnY']
    },
    'VT': {
        'train': ['gzDbaEs1Rlg', 'XzYM3PfTM4w', '98MoyGZKHXc', 'AwmHb44_ouw'],
        'val': ['J0nA4VgnoCo']
    },
    'VU': {
        'train': ['akI8YFjEmUw', 'HT5vyqe0Xaw', 'vdmoEJ5YbrQ', 'xwqBXPGE9pQ'],
        'val': ['sTEELN-vY30']
    }
}

class StartEndDataset(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0,
                 dset_domain=None, cuts_type='single', is_training=True):
        self.is_training=is_training
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        if max_v_l == -1:
            max_v_l = 100000000
        if max_q_l == -1:
            max_q_l = 200
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0

        # Added by me
        self.cuts_type = cuts_type
        self.is_training = is_training

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES

        if self.dset_name == 'hd_epic' or self.dset_name == 'yc2' or self.dset_name == 'hl':
            self.durations = self.compute_durations()
        else:
            self.durations = None

        # data
        self.load_data()
        
        # load specific domain data for tvsum dataset
        if self.dset_name in ['tvsum', 'tvsum_sfc']:
            target_domain = dset_domain
            assert target_domain in ["BK", "BT", "DS", "FM", "GA", "MS", "PK", "PR", "VT", "VU"]

            new_data = []
            for d in self.data:
                if target_domain == d['domain']:
                    new_data.append(d)
            self.data = new_data
            
        # load specific domain data for youtube-hl dataset
        if self.dset_name == 'youtube_uni':
            target_domain = dset_domain
            assert target_domain in ["dog", "gymnastics", "parkour", "skating", "skiing", "surfing"]
            
            new_data = []
            for d in self.data:
                if target_domain == d['domain']:
                    new_data.append(d)
            self.data = new_data    
        
        self.use_glove = False
        self.use_glove = 'vgg' in self.v_feat_dirs[0]

    def compute_durations(self):
        # Done only for HD_EPIC to compute the duration first, which is then needed to make the cuts
        # Note that this is correct for all the datasets we want since all of them have extracted internvideo features with fps 3
        
        durations = {}
        for _feat_dir in self.v_feat_dirs:
            for file_name in os.listdir(_feat_dir):
                if file_name.endswith('.npz'):
                    vid = file_name.replace('.npz', '')
                    _feat_path = join(_feat_dir, file_name)
                    try:
                        feat = np.load(_feat_path)["visual_feats_interm_pooled"]
                        num_frames = feat.shape[0]
                        durations[vid] = num_frames / 3.0
                    except Exception as e:
                        feat = np.load(_feat_path)["visual_feats_pooled"]
                        num_frames = feat.shape[0]
                        durations[vid] = num_frames / 3.0
        return durations

    def _compute_n_gt_merged_window(self, rel_windows_by_qid, clip_data, rel_windows_clip):
        """
        Count how many ground truth windows overlap the current clip window.
        
        Args:
            rel_windows_by_qid: list of tuples [(st1, ed1), (st2, ed2), ...] GT start/end times
            clip_data: dict with 'cut_start' and 'cut_end' indicating window boundaries
        
        Returns:
            int: number of GT windows overlapping current clip window
        """
        n_gt_within_window = 0
        gt_within_windows = []
        cut_start = clip_data['cut_start']
        cut_end = clip_data['cut_end']
        
        for window in rel_windows_by_qid:
            window = window.numpy() if torch.is_tensor(window) else window
            s,e = window
            
            # Check overlap: GT window start before clip end AND GT window end after clip start. The clips are in frame number, so we need to translate to seconds by dividing by 3 (the fps)
            if (s <= (cut_end/3)) and (e >= (cut_start/3)):
                n_gt_within_window += 1
                gt_within_windows.append((s,e))

        
        return n_gt_within_window



    def load_data(self):
        datalist = load_jsonl(self.data_path)
        new_data = []
        cut_mode = self.cuts_type # default single cut if not set


        if self.dset_name == 'hd_epic' or self.dset_name == 'yc2':
            
            # ----------------------
            # Preliminearies
            # Merge all the relevant windows based on their qid to merge them (qid may be shared between independent narrations that share the same underspecific query)
            qid2data_merged = {}
            for d in datalist:
                qid = d["qid"] if "qid" in d else d.get("unique_narration_id", None)
                if qid not in qid2data_merged:
                    qid2data_merged[qid] = []
                
                # Compute the relevant windows in frames
                if 'start_timestamp' in d and 'end_timestamp' in d:
                    relevant_windows = torch.tensor([[d['start_timestamp'], d['end_timestamp']]]) 
                elif "relevant_windows" in d:
                    relevant_windows = torch.tensor(d['relevant_windows'])
                    if relevant_windows.dim() == 1:
                        relevant_windows = relevant_windows.unsqueeze(0)
                else:
                    raise ValueError("No relevant windows in hd_epic data")
                
                qid2data_merged[qid].extend(relevant_windows)
            
        # ----------------------


        for i, data in enumerate(datalist):
            try:
                qid = data["qid"]
            except:
                qid = data.get("unique_narration_id", None)
                data["qid"] = qid

            if self.dset_name == 'hd_epic' or self.dset_name == 'yc2':
                max_len = 500   # clip length in frames
                stride = 500  # frames, for sliding window. Adjust as desired (max_len * 0.66) TODO: This has been changed to not consider sliding windows for now.
  
                if 'start_timestamp' in data and 'end_timestamp' in data:
                    relevant_windows = torch.tensor([[data['start_timestamp'], data['end_timestamp']]]) 
                elif "relevant_windows" in data:
                    relevant_windows = torch.tensor(data['relevant_windows'])
                    if relevant_windows.dim() == 1:
                        relevant_windows = relevant_windows.unsqueeze(0)
                else:
                    raise ValueError("No relevant windows in hd_epic data")
                
                vid_name = data['video_id']
                
                if self.dset_name == 'hd_epic':
                    vid_name = vid_name.split('-')[0] + vid_name
                ctx_l = int(self.durations[vid_name] * 3)  # total frames (fps=3)

                rel_sts = (relevant_windows[:, 0] * 3).tolist() # Set to frames
                rel_eds = (relevant_windows[:, 1] * 3).tolist() # Set to frames

                # ---------------------s
                # This is used to work for augmentations that have a narration_id that map to the merged query (and thus textual features), but we need to preserve the original_narration_id to then evaluate separately
                data['original_qid'] = data.get('original_narration_id', qid)  # To keep track of the original qid in case of multiple cuts
                # ---------------------

                if cut_mode == 'single':
                    # One clip covering all instances (as in your original code)
                    rel_st = int(relevant_windows[:, 0].min() * 3)
                    rel_ed = int(relevant_windows[:, 1].max() * 3)
                    if ctx_l > max_len:
                        min_start = max(0, rel_ed - max_len + 1)
                        max_start = min(rel_st, ctx_l - max_len)
                        if min_start <= max_start:
                            cut_start = np.random.randint(min_start, max_start + 1)
                        else:
                            cut_start = max(0, rel_st - (max_len - (rel_ed - rel_st + 1)) // 2)
                        cut_end = cut_start + max_len
                    else:
                        cut_start = 0
                        cut_end = ctx_l

                    clip_data = data.copy()
                    clip_data['cut_start'] = cut_start
                    clip_data['cut_end'] = cut_end

                    # Adjust relevant windows relative to clip_start and clamp
                    rel_windows_clip = []
                    segs_to_eval = []

                    for st_f, ed_f in zip(rel_sts, rel_eds):
                        overlap_st = max(st_f, cut_start)
                        overlap_ed = min(ed_f, cut_end)
                        if overlap_st < overlap_ed:
                            rel_windows_clip.append([(overlap_st - cut_start) / 3, (overlap_ed - cut_start) / 3])
                            
                            if 'segs_to_evaluate' in data:
                                segs_to_eval.append(data['segs_to_evaluate'][i] == 'true')
                            else:
                                segs_to_eval.append(True)   # If nothing defined, evaluate it
                        
                    clip_data['relevant_windows'] = torch.tensor(rel_windows_clip)  # Now this is set to seconds
                    clip_data['segs_to_eval'] = segs_to_eval
                    
                    duration_sec = (cut_end - cut_start) / 3
                    clip_data['duration'] = duration_sec
                    clip_data['ctx_l'] = max_len
                    clip_data['query'] = clip_data.get('narration', clip_data.get('query', ''))
                    clip_data['vid'] = clip_data['video_id']
                    
                    clip_data['num_gt_merged_window'] = self._compute_n_gt_merged_window(qid2data_merged[qid], clip_data, clip_data['relevant_windows'])
                    new_data.append(clip_data)

                elif cut_mode == 'multi':
                    # Merge overlapping or close windows (within max_len frames. In other words, create groups of relevant windows that will go together in a clip and which fit perfectly inside.
                    # TODO: How do we deal with overlaps but not close windows? This also applies to the code below
                    merged_windows = []
                    sorted_windows = sorted(zip(rel_sts, rel_eds), key=lambda x: x[0])
                    current_start, current_end = sorted_windows[0]
                    for st, ed in sorted_windows[1:]:
                        if st <= current_end + max_len:
                            current_end = max(current_end, ed)
                        else:
                            merged_windows.append((current_start, current_end))
                            current_start, current_end = st, ed
                    merged_windows.append((current_start, current_end))
                    
                    # Generate one cut per merged window ensuring full coverage with randomness
                    for mw_st, mw_ed in merged_windows:
                        clip_len = max_len
                        earliest_start = max(0, mw_ed - clip_len)      # earliest start so full window fits in clip
                        latest_start = min(mw_st, ctx_l - clip_len)    # latest start position
                        
                        if earliest_start > latest_start:
                            clip_st = earliest_start
                        else:
                            # Randomize clip start position for positional variation
                            clip_st = np.random.randint(earliest_start, latest_start + 1)
                        clip_ed = clip_st + clip_len
                        
                        # Find all relevant windows (fully or partially) overlapping with this clip
                        rel_windows_clip = []
                        segs_to_eval = []
                        
                        for st_f, ed_f in zip(rel_sts, rel_eds):
                            # This overlap guarantees to take as an instance the part that overlaps
                            # Leave this as it is for now??
                            overlap_st = max(st_f, clip_st)
                            overlap_ed = min(ed_f, clip_ed)
                            if overlap_st < overlap_ed:
                                # Convert overlap to clip-relative seconds
                                rel_windows_clip.append([(overlap_st - clip_st) / 3, (overlap_ed - clip_st) / 3])
                                
                                if 'segs_to_evaluate' in data:
                                    segs_to_eval.append(data['segs_to_evaluate'][i] == 'true')
                                else:
                                    segs_to_eval.append(True)   # If nothing defined, evaluate it
                                
                                
                        # Create clip data only if there's at least one GT window overlap
                        if rel_windows_clip:
                            clip_data = data.copy()
                            clip_data['cut_start'] = clip_st
                            clip_data['cut_end'] = clip_ed
                            clip_data['relevant_windows'] = torch.tensor(rel_windows_clip)
                            clip_data['segs_to_eval'] = segs_to_eval

                            clip_data['duration'] = (clip_ed - clip_st) / 3
                            clip_data['ctx_l'] = max_len
                            clip_data['query'] = clip_data.get('narration', clip_data.get('query', ''))
                            clip_data['vid'] = clip_data['video_id']
                            clip_data['qid'] = qid
                            
                            # Important note: So all the cuts of this sliding window share the same qid (which is the unique narration id)
                            clip_data['num_gt_merged_window'] = self._compute_n_gt_merged_window(qid2data_merged[qid], clip_data, rel_windows_clip)
                            new_data.append(clip_data)
                            
                elif cut_mode == 'sliding':
                    # Systematic sliding windows over full video with overlap stride
                    start_positions = list(range(0, max(1, ctx_l - max_len + 1), stride))

                    if start_positions[-1] + max_len < ctx_l:
                        start_positions.append(start_positions[-1] + stride)    # Create a final window that covers the very end
                        
                        # This considerably improves the mAP metric (TODO: Check)
                        # Using it however messes up a bit the window-level metrics
                        #start_positions.append(ctx_l - max_len) 

                    for clip_st in start_positions:
                        clip_ed = clip_st + max_len

                        clip_data = data.copy()
                        clip_data['cut_start'] = clip_st
                        clip_data['cut_end'] = clip_ed

                        rel_windows_clip = []
                        segs_to_eval = []
                        for st_f, ed_f in zip(rel_sts, rel_eds):
                            overlap_st = max(st_f, clip_st)
                            overlap_ed = min(ed_f, clip_ed)
                            if overlap_st < overlap_ed:
                                rel_windows_clip.append([(overlap_st - clip_st) / 3, (overlap_ed - clip_st) / 3])

                                # Optionally: Get also wheather this segments should actually be evaluated
                                if 'segs_to_evaluate' in data:
                                    segs_to_eval.append(data['segs_to_evaluate'][i] == 'true')
                                else:
                                    segs_to_eval.append(True)   # If nothing defined, evaluate it

                        clip_data['relevant_windows'] = torch.tensor(rel_windows_clip)

                        # Optionally, we might have segs_to_evaluate (which defines which concrete of these segments should be actually evaluated)
                        clip_data['segs_to_eval'] = segs_to_eval
                        
                        duration_sec = (clip_ed - clip_st) / 3
                        clip_data['duration'] = duration_sec
                        clip_data['ctx_l'] = max_len
                        clip_data['query'] = clip_data.get('narration', clip_data.get('query', ''))
                        clip_data['vid'] = clip_data['video_id']
                        
                        # Skip if it does not contain a relevant window
                        if not rel_windows_clip:
                            # Optionally: print(f"Skipping window [{clip_st}:{clip_ed}]: no relevant intervals")
                            continue
                        
                        clip_data['num_gt_merged_window'] = self._compute_n_gt_merged_window(qid2data_merged[qid], clip_data, rel_windows_clip)    
                        
                        new_data.append(clip_data)
                else:
                    raise ValueError(f"Unknown cut_mode: {cut_mode}")

            else:
                
                # ---------------------s
                # This is used to work for augmentations that have a narration_id that map to the merged query (and thus textual features), but we need to preserve the original_narration_id to then evaluate separately
                data['original_qid'] = data.get('original_narration_id', qid)  # To keep track of the original qid in case of multiple cuts
                data['num_gt_merged_window'] = len(data.get('relevant_windows', [])) if self.load_labels else 0
                
                if self.durations is not None:
                    data['duration'] = float(self.durations[data['video_id']])
                # ---------------------

                # Other datasets keep original behaviour (single sample)
                new_data.append(data)

        if self.data_ratio != 1:
            n_examples = int(len(new_data) * self.data_ratio)
            new_data = new_data[:n_examples]
            logger.info(f"Using {self.data_ratio * 100}% of the data: {n_examples} examples")

        self.data = new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]

        model_inputs = dict()

        if "qid" in meta:
            qid = meta["qid"]
        elif "unique_narration_id" in meta:
            qid = meta["unique_narration_id"]
            meta['qid'] = qid
        model_inputs["query_feat"] = self._get_query_feat_by_qid(qid, meta)  # (Dq, ) or (Lq, Dq)
            
        if self.use_video:
            if "vid" in meta:
                vid = meta["vid"]
            elif "video_id" in meta:
                vid = meta["video_id"]
            else:
                raise NotImplementedError
            model_inputs["video_feat"] = self._get_video_feat_by_vid(vid, meta=meta)  # (Lv, Dv)
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l
        
        
        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef
                
        if self.dset_name == 'hd_epic' or self.dset_name == 'yc2':
            relevant_windows = torch.tensor(meta['relevant_windows'])
            duration = meta['duration']
            ctx_l = meta['ctx_l']
            
            # ----------------
            # Optional to do gt replication. Added by me
            if False and self.is_training:
                model_inputs["video_feat"], relevant_windows = self._replicate_gt_segments(
                    model_inputs["video_feat"],
                    relevant_windows,
                    duration,
                    num_replications=5,
                )
            # ----------------
                  
            if len(relevant_windows) == 0:
                # If no relevant windows, create dummy labels
                model_inputs["span_labels"] = torch.tensor([[0, 0]])  # Dummy span label
                model_inputs["saliency_pos_labels"] = [0,0] # We need two values
                model_inputs["saliency_neg_labels"] = [0,0] # We need two values
                model_inputs["saliency_all_labels"] = np.zeros(ctx_l)
            else:
                model_inputs["span_labels"] = self.get_span_labels(relevant_windows, ctx_l, clip_len=1/3)
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_sub_as_query(relevant_windows[0], duration, ctx_l, 2)  # only one gt
                        
            meta['duration'] = duration
            meta['query'] = meta['narration']
            meta['vid'] = meta['video_id']
        elif self.dset_name == 'synopground':
            relevant_windows = torch.tensor(meta['relevant_windows'])   # The relevant windows are also in frames not seconds
            duration = meta['duration'] # In this case this duration is in frames
            
            # Important note: The FPS are actually 3, but since the GT and all the annos are frame-level, we can just pretend that FPS=1, so that the rest of the code runs as if every frame is a second.
            model_inputs["span_labels"] = self.get_span_labels(relevant_windows, ctx_l, clip_len=1)
            model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_sub_as_query(relevant_windows[0], duration, ctx_l, 2)  # only one gt  
        else:
            if "relevant_windows" in meta: ## For Qvhighlights test set
                model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)  # (#windows, 2)
                if self.dset_name in ['charadesSTA', 'tacos', 'activitynet']: ## charades, tacos, nlq
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], meta["duration"], ctx_l)  # only one gt
                elif self.dset_name in ['nlq']:
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], meta["duration"], ctx_l, 2)  # only one gt
                elif "subs_train" not in self.data_path:
                    if "relevant_clip_ids" in meta and "saliency_scores" in meta:
                        model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                            self.get_saliency_labels_all(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
                    else:
                        relevant_windows = torch.tensor(meta['relevant_windows'])
                        
                        if 'duration' in meta:
                            duration = float(meta['duration'])
                        else:
                            duration = float(self.durations[meta['video_id']]) 
                        model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = self.get_saliency_labels_sub_as_query(relevant_windows[0], duration, ctx_l, 2)  # only one gt
                
                else:
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs[
                        "saliency_all_labels"] = \
                        self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], meta["duration"], ctx_l)  # only one gt

        if 'qvhighlight' in self.data_path:
            model_inputs["relevant_clip_ids"] = meta["relevant_clip_ids"]
            
        if "vid" in meta:
            model_inputs["vid"] = meta["vid"]
        elif "video_id" in meta:
            model_inputs["vid"] = meta["video_id"]
        else:
            raise NotImplementedError
        
        if "qid" in meta:
            model_inputs["qid"] = meta["qid"]
        elif "unique_narration_id" in meta:
            model_inputs["qid"] = meta["unique_narration_id"]
            meta["qid"] = meta["unique_narration_id"]
        if "query" not in meta:
            meta["query"] = meta['narration']
        
        return dict(meta=meta, model_inputs=model_inputs)

    def get_saliency_labels_sub_as_query(self, gt_window, duration, ctx_l, max_n=2):
        clip_len = duration / ctx_l
        gt_st = int(gt_window[0] / clip_len)
        gt_ed = max(0, min(int(gt_window[1] / clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed + 1), k=max_n)
        else:
            if self.dset_name == 'nlq':
                pos_clip_indices = [gt_st] * 2
            else:
                pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l))
        try:
            neg_clip_indices = random.sample(neg_pool, k=max_n)
        except:
            neg_clip_indices = pos_clip_indices

        # For charades_sta
        score_array = np.zeros(ctx_l)
        score_array[gt_st:gt_ed + 1] = 1

        return pos_clip_indices, neg_clip_indices, score_array
        

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels_all(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        try:
            # score_array = [min(agg_scores[idx], ctx_l-1) for idx in range(ctx_l)]
            score_array = np.zeros(ctx_l)
            for idx in range(len(rel_clip_ids)):
                if rel_clip_ids[idx] >= ctx_l:
                    score_array_new = np.zeros(ctx_l + 1)
                    score_array_new[:ctx_l] = score_array
                    score_array = score_array_new
                score_array[rel_clip_ids[idx]] = agg_scores[idx]
        except:
            score_array = np.zeros(ctx_l)   # TODO: Added by me
            
        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels_all_tvsum(self, labels, ctx_l, max_n=1, add_easy_negative=False):
        
        agg_scores = np.sum(labels - np.ones_like(labels), axis=-1)[:ctx_l] # start from 1, so minus 1
        score_array = agg_scores / 80 * 12
        sort_indices = np.argsort(agg_scores)  # increasing

        hard_pos_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels_all_youtube(self, labels, ctx_l, max_n=1, add_easy_negative=False):
        
        # Youtube-hl only have binary score
        agg_scores = np.array(labels)[:, 0] # (L, 1) --> (L, )
        score_array = agg_scores * 1
        
        sort_indices = np.argsort(agg_scores)  # increasing

        hard_pos_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices, score_array
    
    
    def get_span_labels(self, windows, ctx_l, clip_len=None):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if clip_len is None:
            clip_len = self.clip_len
        
        if len(windows) > self.max_windows and self.max_windows != -1:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / clip_len), min(int(w[1] / clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid, meta=None):
        if self.dset_name == 'tvsum':
            q_feat = np.load(join(self.q_feat_dir, "{}.npz".format(qid))) # 'token', 'text'
            return torch.from_numpy(q_feat['token'])
        # youtube-hl
        elif self.dset_name == 'youtube_uni':
            q_feat = np.load(join(self.q_feat_dir, "{}.npz".format(qid)))
            return torch.from_numpy(q_feat['last_hidden_state'])
        
        elif self.dset_name in ['tacos', 'nlq']:
            q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
            q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
            if self.q_feat_type == "last_hidden_state":
                q_feat = q_feat[:self.max_q_l]
            if self.normalize_t:
                q_feat = l2_normalize_np_array(q_feat)
            if self.txt_drop_ratio > 0:
                q_feat = self.random_drop_rows(q_feat)
        elif self.dset_name in ['hd_epic', 'hl', 'yc2']:
            q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
            q_feat = np.load(q_feat_path)['text_feats'].astype(np.float32)  # Select the last one
        elif self.dset_name == 'synopground':
            drama_id, video_id = qid.split('-')
            q_feat_path = join(self.q_feat_dir, "%s_%s.pkl"%(drama_id, video_id))
            n_paragraph = meta['paragraph_number']

            q_feat = pickle.load(open(q_feat_path, 'rb'))[n_paragraph][0]
        else:
            raise NotImplementedError
   
   
        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

   
    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_video_feat_by_vid(self, vid, meta=None):
        if self.dset_name == 'tvsum':
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                _feat_path = join(_feat_dir, f"{vid}_rgb.npy")
                _feat_rgb = np.load(_feat_path)[:self.max_v_l].astype(np.float32)

                _feat_path = join(_feat_dir, f"{vid}_opt.npy")
                _feat_opt = np.load(_feat_path)[:self.max_v_l].astype(np.float32)
                
                _feat = np.concatenate([_feat_rgb, _feat_opt], axis=-1)
                # _feat = _feat_rgb
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)

        elif self.dset_name == 'youtube_uni':
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                # Only single npz files per directory
                try:
                    _feat_path = join(_feat_dir, f"{vid}.npz")
                    _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                except:
                    _feat_path = join(_feat_dir, f"{vid}.npy")
                    _feat = np.load(_feat_path)[:self.max_v_l].astype(np.float32)
                
                # _feat = _feat_rgb
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list] # TODO do we need to cut the length over the min_len?
            v_feat = np.concatenate(v_feat_list, axis=1)
        elif self.dset_name == 'hl':
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                _feat_path = join(_feat_dir, f"{vid}.npz")
                
                try:
                    _feat = np.load(_feat_path)["visual_feats_interm_pooled"][:self.max_v_l, -1].astype(np.float32) # Select only the last representation (This is the one we should be using)                
                except Exception as e:
                    print(e)
                    print(f"Error loading {vid} from {_feat_path}")
                    import time
                    time.sleep(10)
                v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)
        elif self.dset_name == 'hd_epic' or self.dset_name == 'yc2':
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                if self.dset_name == 'hd_epic':
                    participant = vid.split('-')[0]
                    _feat_path = join(_feat_dir, f"{participant}{vid}.npz")
                else:
                    _feat_path = join(_feat_dir, f"{vid}.npz")
                    
                
                _feat = np.load(_feat_path)["visual_feats_interm_pooled"][:self.max_v_l, -1].astype(np.float32) # Select only the last representation (This is the one we should be using)                
                
                # Recover the cut start and end
                cut_start = int(meta['cut_start'])
                cut_end = int(meta['cut_end'])
                
                _feat = _feat[cut_start:cut_end]
            
                v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)
        elif self.dset_name == 'synopground':
            drama_id, video_id = meta['qid'].split('-')
            # slowfast features
            video_feat = np.load(os.path.join(self.v_feat_dirs[0], "%s_%s.npy"%(drama_id, video_id)))
            video_feat = torch.from_numpy(video_feat) if len(video_feat) <= 2500 else self._align_feat(video_feat, 2500)    # Allows to reformat the entire video length to a maximum length of 2500
            # clip features
            clip_feat = np.load(os.path.join(self.v_feat_dirs[1], "%s_%s.npy"%(drama_id, video_id)))
            clip_feat = self._align_feat(clip_feat, video_feat.shape[0])  # [len(video_feat), c]
            # concatenated features
            v_feat = torch.cat((video_feat, clip_feat), dim=-1)
       
        if not torch.is_tensor(v_feat):
            return torch.from_numpy(v_feat)  # (Lv, D)
        else:
            return v_feat
       
    def _align_feat(self, in_feat, tgt_len):
        # This function is for Synopground features (implemented by the authors)
        src_len, feat_dim = in_feat.shape
        out_feat = F.adaptive_avg_pool1d(torch.from_numpy(in_feat).float().transpose(0, 1).contiguous().view(1, feat_dim, src_len), tgt_len)  # [1, feat_dim, tgt_len]
        out_feat = out_feat[0].transpose(0, 1).contiguous()  # [tgt_len, feat_dim]
        return out_feat

# Optionally added to replicat GT into other parts of the video
    def _replicate_gt_segments(self, video_feat, gt_windows, duration, num_replications=1):
        """
        Replicate GT segments to random background locations without overlapping with any GT.

        Args:
            video_feat: Tensor [T, D] original video features
            gt_windows: Tensor [#GT, 2] in seconds
            duration: total video duration (seconds)
            num_replications: int, number of replications per GT
            iou_thresh: maximum allowed IoU with any GT for replication

        Returns:
            new_video_feat: Tensor [T, D] video features with replicated segments
            new_gt_windows: Tensor [(#GT + num_replications*#GT), 2] replicated GT coordinates
        """

        T, D = video_feat.shape
        new_video_feat = video_feat.clone()
        new_gt_windows = gt_windows.clone().tolist()

        def temporal_iou(seg1, seg2):
            """Compute temporal IoU between two segments [start, end]."""
            s1, e1 = seg1
            s2, e2 = seg2
            inter = max(0, min(e1, e2) - max(s1, s2))
            union = max(e1, e2) - min(s1, s2)
            return inter / union if union > 0 else 0.0

        for gt in gt_windows:
            start, end = gt
            length = end - start
            scale = 2  # or 2.0

            if random.random() < 0.5:
                # With 50% probability, we don't do anything
                rand_num_replications = 0
            else:
                # Choose a number between 1 and num_replications
                rand_num_replications = random.randint(1, num_replications + 1)

            for _ in range(rand_num_replications):
                # Try to find a non-overlapping background slot
                for _ in range(10):  # try up to 10 times
                    bg_start = random.uniform(0, duration - length*scale)
                    bg_end = bg_start + length
                    center = (bg_start + bg_end) / 2
                    scaled_half_len = (bg_end - bg_start) * scale / 2
                    scaled_bg_start = max(0, center - scaled_half_len)
                    scaled_bg_end = min(duration, center + scaled_half_len)
                    

                    # Check IoU with all existing GT windows
                    if all(temporal_iou([scaled_bg_start, scaled_bg_end], existing) <= 0.0 for existing in new_gt_windows):
                        break
                else:
                    # Could not find non-overlapping slot, skip replication
                    continue


                # Get the new coordinates in frames where the video features of the replicated action should be pasted
                f_start = int(scaled_bg_start / duration * T)  
                f_end = int(scaled_bg_end / duration * T)
                
                # Scale the GT coordinates including also some near-boundary information
                center = (start + end) / 2
                half_len = (end - start) * scale / 2
                scaled_start = max(0, center - half_len)
                scaled_end = min(duration, center + half_len)
                
                # Get the coordinates in frames of the original GT video features
                gt_f_start = int(scaled_start / duration * T)  
                gt_f_end = int(scaled_end / duration * T)  

                # Replicate the features including also some near-boundary information
                rep_feat = video_feat[gt_f_start:gt_f_end+1]
                rep_feat_len = f_end - f_start + 1
                
                if rep_feat_len > 0:
                    rep_feat = F.interpolate(
                        rep_feat.T.unsqueeze(0), size=rep_feat_len, mode="linear", align_corners=False
                    ).squeeze(0).T
                    new_video_feat[f_start:f_end+1] = rep_feat

                    new_gt_windows.append([bg_start, bg_end])
                

        return new_video_feat, torch.tensor(new_gt_windows)


def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        if k == "saliency_all_labels":
            pad_data, mask_data = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=np.float32, fixed_length=None)
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue
        if k == 'qid':
            batched_data[k] = [e["model_inputs"][k] for e in batch]
            continue
        if k == 'vid':
            batched_data[k] = [e["model_inputs"][k] for e in batch]
            continue
        batched_data[k] = pad_sequences_1d(
            [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
    
    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
        vid=batched_model_inputs["vid"],
        qid=batched_model_inputs["qid"],
    )
    targets = {}

    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)
        targets["relevant_clips"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)
    targets = None if len(targets) == 0 else targets
    return model_inputs, targets
