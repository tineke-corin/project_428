from settings import settings
from app.services.utils import Point, BoundingBox, is_within_box, intersection_over_union, point_from_normalised, box_from_normalised

"""
When IoU > threshold (50), it is a TP. If IoU < threshold it is a FP.
If there is a predicted BB but it does not intersect with any ground truth BB then that is also a FP.
FN is when there is an annotated face/VLP that the model fails to detect.
precision at threshold 50: p(50) = TP/(TP+FP)
Recall = TP/(TP+FN)
"""

def get_statistics(ground_truths, detections):
  statistics = { 'fp': 0, 'tp': 0, 'fn': 0 }  
  matches = []  
  for d_index, d in enumerate(detections):
    best_iou = 0
    best_ground_truth_index = -1
      
    for truth_index, truth in enumerate(ground_truths):
      if truth_index in matches:
        # we already matched this truth to a detection
        continue

      if d.contains_point(truth.start_point) and d.contains_point(truth.end_point):
        # the detection fully contains the ground truth. Normally IoU would depend on
        # the detection not being too much larger than the ground truth, but for our
        # purposes we call this a TP regardless of the ratio.
        statistics['tp'] += 1
        matches.append(truth_index)
        continue

      iou = intersection_over_union(d, truth)
      if iou > best_iou:
        # this is the best match found so far for this GT
        best_iou = iou
        best_ground_truth_index = truth_index

    if best_iou >= settings.iou_threshold:
      # this is a TP match
      statistics['tp'] += 1
      matches.append(best_ground_truth_index)
    else:
      statistics['fp'] += 1

  statistics['fn'] = len(ground_truths) - len(matches)
  return statistics

def get_precision_and_recall(statistics):
  total_tp = sum([ x['tp'] for x in statistics ])
  total_fp = sum([ x['fp'] for x in statistics ])
  total_fn = sum([ x['fn'] for x in statistics ])
  precision = (total_tp / ( total_tp + total_fp ))
  recall = (total_tp / ( total_tp + total_fn ))
  f1 = 2 * ( (precision * recall) / (precision + recall ))
  return precision, recall, f1
