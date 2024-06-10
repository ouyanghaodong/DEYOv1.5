import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops


class DEYODetectionValidator(DetectionValidator):
    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        bs, nq, nd = preds[0].shape
        num_select = nq
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        # bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        topk_values, topk_indexes = torch.topk(scores.reshape(scores.shape[0], -1), num_select, dim=1)
        topk_boxes = topk_indexes // scores.shape[2]
        labels = topk_indexes % scores.shape[2]
        bboxes = torch.gather(bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        scores = topk_values
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score = scores[i]
            cls = labels[i]
            # Do not need threshold for evaluation as only got 300 boxes here
            # idx = score > self.args.conf
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  # filter
            # Sort by confidence to correctly get internal metrics
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred  # [idx]
        return outputs