## Introduction

DEYO and YOLOv10 both utilize a one-to-many branch during the training process to acquire sufficient supervisory signals and truncate the gradients of the one-to-one branch to enhance its performance. In the paper "[DEYOv3: DETR with YOLO for Real-time Object Detection](https://arxiv.org/abs/2309.11851)," we introduced Step-by-step training and thoroughly discussed the importance of the one-to-many branch in obtaining adequate supervision for the backbone and neck. Truncating the gradients of the one-to-one branch is a key technical improvement we made to DEYOv3 in the paper "[DEYO: DETR with YOLO for End-to-End Object Detection](https://arxiv.org/abs/2402.16370)." This clearly illustrates that during the training process, providing thorough supervision to the backbone and neck through the one-to-many branch, coupled with a gradient truncation mechanism that fundamentally eliminates the negative impact of one-to-one matching instability on network performance, is essential for ensuring the good performance of the one-to-one branch.

The training strategy of YOLOv10, although performing well on COCO, may face the following challenges when training on custom datasets compared to DEYO:

1. There is a difference in the convergence speed between the one-to-many branch and the one-to-one branch, which may lead to overfitting in the one-to-many branch while the one-to-one branch has not yet fully converged.

2. The backbone and neck supervised by the one-to-many branch may not adapt well to the one-to-one branch with a pure convolutional structure, resulting in a significant difference in the final convergence accuracy between the one-to-one and one-to-many branches.

We employ step-by-step training and have replaced the Hungarian Matching with TAL. By decoupling the training of these two branches, we effectively address the issue of inconsistent convergence speeds between the two. Additionally, we have developed DEYOv1.5 to prevent significant accuracy loss in certain situations where a pure convolutional structure is directly used for the one-to-one branch. DEYOv1.5 introduces the Scale-adaptive Self-Attention proposed in SparseBEV. This method allows us to maintain performance loss within a controllable range without relying on Deformable Attention, while also avoiding the use of the grid_sample operator, making the model more deployable. Furthermore, during the second phase of training, we freeze the bounding box head of the one-to-many branch and use it directly for the prediction of the bounding box in the one-to-one branch. This strategy significantly enhances the model's performance when dealing with situations where there is a large difference in the final convergence accuracy between the one-to-one and one-to-many branches. DEYOv1.5 maintains high precision while only sacrificing 10% of the speed compared to the one-to-one branch using pure convolutions.


## Models
| Model | Epoch | End-to-End | $AP^{val}$ | $AP^{val}_{50}$ | Params(M) | FLOPs(G) | T4 TRT FP16(FPS) |
|:------|:-----:|:-----------:|:----------:|:---------------:|:---------:|:--------:|:---------------:|
| YOLOv8-N | --  | ✔ | --   | --   | 3.2  | 8.7   | 554 | 
| YOLOv10-N | -- | ✔ | 38.5 | 53.8 | 2.3  | 6.7   | 538 | 
| YOLOv10-N | -- | ✘ | 39.4 | 55.0 | 2.3  | 6.7   | --  |
| YOLOv9-C  | -- | ✔ | 51.9 | 68.7 | 25.3 | 102.7 | 155 |
| YOLOv9-C  | -- | ✘ | 52.9 | 69.8 | 25.3 | 102.7 | --  | 
| YOLOv9-E  | -- | ✔ | 54.6 | 71.4 | 57.4 | 189.5 | 65  |
| YOLOv9-E  | -- | ✘ | 55.1 | 72.2 | 57.4 | 189.5 | --  |
| DEYO-tiny | 96 | ✔ | 37.6 | 52.8 | 4.3  | 7.6   | 487 |
| [DEYOv1.5-N](https://github.com/ouyanghaodong/DEYOv1.5/releases/download/v0.1/deyov1.5n.pt) | 144 | ✔ | 39.5 | 55.7 | 3.1  | 7.2   | 501 | 
| [DEYOv1.5-C](https://github.com/ouyanghaodong/DEYOv1.5/releases/download/v0.1/deyov1.5c.pt) | 72  | ✔ | 52.6 | 69.5 | 26.6 | 87.4  | 135 |
| [DEYOv1.5-E](https://github.com/ouyanghaodong/DEYOv1.5/releases/download/v0.1/deyov1.5e.pt) | 72  | ✔ | 55.0 | 71.9 | 58.7 | 174.2 | 63  |

##### Note: We are using the YOLOv9 implemented by ultralytics, which has a slight difference in accuracy compared to the original version.

## Install
```bash
pip install ultralytics
```

## Step-by-step Training

#### Frist Training Stage
Unlike YOLOv10, DEYO selects the backbone and neck most suitable for the one-to-many branch, rather than those most suitable for the one-to-one branch, achieving true alignment with the accuracy of the one-to-many branch.

Replace `ultralytics/engine/trainer.py` with `trainer.py`

```python
from ultralytics import YOLO

# Train from Scratch
model = YOLO("cfg/models/v10/yolov10n.yaml")

# Use the model
model.train(data = "coco.yaml", epochs = 500, scale = 0.5, mixup = 0, copy_paste = 0)

# Train from Scratch
model = YOLO("cfg/models/v9/yolov9c.yaml")

# Use the model
model.train(data = "coco.yaml", epochs = 500, scale = 0.9, mixup = 0.15, copy_paste = 0.3)

# Train from Scratch
model = YOLO("cfg/models/v9/yolov9e.yaml")

# Use the model
model.train(data = "coco.yaml", epochs = 500, scale = 0.9, mixup = 0.15, copy_paste = 0.3)
```

#### Second Training Stage

Please note that if you directly adopt a model pre-trained on the COCO dataset, you may not achieve the best results, as we have set the one-to-many branch to a frozen state, which means the one-to-many branch will not undergo further fine-tuning optimization based on your dataset. You will need to fine-tune the model pre-trained on the COCO dataset during the First Training Stage.

If you have made changes to the `ultralytics/engine/trainer.py` during the frist training stage, please revert it.

```python
from ultralytics import DEYO

# Load a model
model = DEYO("cfg/models/deyo/deyov1.5n.yaml")
model.load("best-n.pt")

# Use the model
model.train(data = "coco.yaml", epochs = 144, lr0 = 0.0001, lrf = 0.0001, weight_decay = 0.0001, optimizer = 'AdamW', warmup_epochs = 0, mosaic = 0, scale = 0.5, mixup = 0, copy_paste = 0, freeze = 23)

# Load a model
model = DEYO("cfg/models/deyo/deyov1.5c.yaml")
model.load("best-c.pt")

# Use the model
model.train(data = "coco.yaml", epochs = 72, lr0 = 0.0001, lrf = 0.0001, weight_decay = 0.0001, optimizer = 'AdamW', warmup_epochs = 0, mosaic = 0, scale = 0.9, mixup = 0.15, copy_paste = 0.3, freeze = 22)

# Load a model
model = DEYO("cfg/models/deyo/deyov1.5e.yaml")
model.load("best-e.pt")

# Use the model
model.train(data = "coco.yaml", epochs = 72, lr0 = 0.0001, lrf = 0.0001, weight_decay = 0.0001, optimizer = 'AdamW', warmup_epochs = 0, mosaic = 0, scale = 0.9, mixup = 0.15, copy_paste = 0.3, freeze = 22)
```

## Multi GPUs
The multi-GPU setup for the first stage of training is consistent with YOLOv8. For the second stage of training, you need to follow the following steps:

Replace `ultralytics/engine/trainer.py` with our modified `ddp/trainer.py`
```bash
rm -rf Path/ultralytics
cp -r ultralytics Path/  # Path：The location of the ultralytics package
```

```python
import torch
from ultralytics import DEYO

# Load a model
model = DEYO("cfg/models/deyo/deyov1.5e.yaml")
model.load("best-e.pt")
torch.save({"epoch":-1, "model": model.model.half(), "optimizer":None}, "init.pt")
model = DEYO("init.pt")

# Use the model
model.train(data = "coco.yaml", epochs = 72, lr0 = 0.0001, lrf = 0.0001, weight_decay = 0.0001, optimizer = 'AdamW', warmup_epochs = 0, mosaic = 0, scale = 0.9, mixup = 0.15, copy_paste = 0.3, freeze = 22, device = '0, 1, 2, 3, 4, 5, 6, 7')
```

## Benchmark
You can follow the method we used in  [DEYO](https://github.com/ouyanghaodong/DEYO).

## License
This project builds heavily off of [ultralytics](https://github.com/ultralytics/ultralytics). Please refer to their original licenses for more details.

## Citation
If you use `DEYOv1.5` in your work, please use the following BibTeX entries:
```
@article{Ouyang2023DEYOv3,
  title={DEYOv3: DETR with YOLO for Real-time Object Detection},
  author={Haodong Ouyang},
  journal={ArXiv},
  year={2023},
  volume={abs/2309.11851},
}

@article{Ouyang2024DEYO,
  title={DEYO: DETR with YOLO for End-to-End Object Detection},
  author={Haodong Ouyang},
  journal={ArXiv},
  year={2024},
  volume={abs/2402.16370},
}
```
