from ..yolo import YOLO
from ultralytics.nn.tasks import DEYODetectionModel
from .val import DEYODetectionValidator
from .predict import DEYODetectionPredictor
from .train import DEYODetectionTrainer

class DEYO(YOLO):
    def __init__(self, model="deyo-n.pt") -> None:
        super().__init__(model=model, task="detect")
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DEYODetectionModel,
                "trainer": DEYODetectionTrainer,
                "validator": DEYODetectionValidator,
                "predictor": DEYODetectionPredictor,
            },
        }