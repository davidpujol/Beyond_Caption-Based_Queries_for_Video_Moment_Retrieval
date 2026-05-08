"""
Non-Maximum Suppression for video proposals.
"""
import math

def compute_temporal_iou(pred, gt):
    """ deprecated due to performance concerns
    compute intersection-over-union along temporal axis
    Args:
        pred: [st (float), ed (float)]
        gt: [st (float), ed (float)]
    Returns:
        iou (float):

    Ref: https://github.com/LisaAnne/LocalizingMoments/blob/master/utils/eval.py
    """
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])  # not the correct union though
    if union == 0:
        return 0
    else:
        return 1.0 * intersection / union


def temporal_nms(predictions, nms_thd, max_after_nms=100, min_conf=0.0):
    """
    Args:
        predictions: list(sublist), each sublist is [st (float), ed(float), score (float)],
            note larger scores are better and are preserved. For metrics that are better when smaller,
            please convert to its negative, e.g., convert distance to negative distance.
        nms_thd: float in [0, 1]
        max_after_nms:
    Returns:
        predictions_after_nms: list(sublist), each sublist is [st (float), ed(float), score (float)]
    References:
        https://github.com/wzmsltw/BSN-boundary-sensitive-network/blob/7b101fc5978802aa3c95ba5779eb54151c6173c6/Post_processing.py#L42
    """
    
    # Filter out low-confidence predictions
    predictions = [p for p in predictions if p[2] >= min_conf]
    if len(predictions) == 0:
        return []

    if len(predictions) == 1:  # only has one prediction, no need for nms
        return predictions

    predictions = sorted(predictions, key=lambda x: x[2], reverse=True)  # descending order

    tstart = [e[0] for e in predictions]
    tend = [e[1] for e in predictions]
    tscore = [e[2] for e in predictions]
    rstart = []
    rend = []
    rscore = []
    while len(tstart) > 1 and len(rscore) < max_after_nms:  # max 100 after nms
        idx = 1
        while idx < len(tstart):  # compare with every prediction in the list.
            if compute_temporal_iou([tstart[0], tend[0]], [tstart[idx], tend[idx]]) > nms_thd:
                # rm highly overlapped lower score entries.
                tstart.pop(idx)
                tend.pop(idx)
                tscore.pop(idx)
                # print("--------------------------------")
                # print(compute_temporal_iou([tstart[0], tend[0]], [tstart[idx], tend[idx]]))
                # print([tstart[0], tend[0]], [tstart[idx], tend[idx]])
                # print(tstart.pop(idx), tend.pop(idx), tscore.pop(idx))
            else:
                # move to next
                idx += 1
        rstart.append(tstart.pop(0))
        rend.append(tend.pop(0))
        rscore.append(tscore.pop(0))

    if len(rscore) < max_after_nms and len(tstart) >= 1:  # add the last, possibly empty.
        rstart.append(tstart.pop(0))
        rend.append(tend.pop(0))
        rscore.append(tscore.pop(0))

    predictions_after_nms = [[st, ed, s] for s, st, ed in zip(rscore, rstart, rend)]
    return predictions_after_nms



# Added by me
def temporal_soft_nms(predictions, nms_thd=0.7, max_after_nms=100, min_conf=0.0, sigma=0.5, ):
    """
    Soft-NMS for temporal segments.

    Args:
        predictions: list of [st, ed, score]
        sigma: float, controls the Gaussian decay (default=0.5)
        iou_thresh: IoU threshold for linear variant (still used for Gaussian to skip negligible overlaps)
        max_after_nms: int, number of proposals to keep
        min_conf: float, minimum score to keep a proposal
    Returns:
        list of [st, ed, updated_score]
    """

    # Filter low confidence
    predictions = [p for p in predictions if p[2] >= min_conf]
    if len(predictions) <= 1:
        return predictions

    # Sort by score (descending)
    predictions = sorted(predictions, key=lambda x: x[2], reverse=True)

    tstart = [p[0] for p in predictions]
    tend   = [p[1] for p in predictions]
    tscore = [p[2] for p in predictions]

    keep = []

    while len(tscore) > 0 and len(keep) < max_after_nms:
        # 1. Pick top scoring segment
        max_idx = max(range(len(tscore)), key=lambda i: tscore[i])
        max_st, max_ed, max_score = tstart[max_idx], tend[max_idx], tscore[max_idx]
        keep.append([max_st, max_ed, max_score])

        # 2. Remove it from the candidate list
        tstart.pop(max_idx)
        tend.pop(max_idx)
        tscore.pop(max_idx)

        # 3. Decay the scores of the rest
        for i in range(len(tscore)):
            iou = compute_temporal_iou([max_st, max_ed], [tstart[i], tend[i]])
            if iou > 0:
                # Gaussian decay (preferred)
                tscore[i] *= math.exp(- (iou ** 2) / sigma)
                # Optional: linear decay version
                # if iou > iou_thresh:
                #     tscore[i] *= (1 - iou)
        
        # 4. Re-sort descending
        order = sorted(range(len(tscore)), key=lambda i: tscore[i], reverse=True)
        tstart = [tstart[i] for i in order]
        tend   = [tend[i] for i in order]
        tscore = [tscore[i] for i in order]

    # Return top after NMS
    return keep[:max_after_nms]
