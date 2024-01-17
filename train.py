import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
# from tensorboardX import SummaryWriter
from detectron2.data.datasets import register_coco_instances
# from detectron2.checkpoint import DetectionCheckpointer
from detectron2.layers.losses import diou_loss
# from detectron2.layers.nms import cluster_nms
# Register your custom dataset
# register_coco_instances("my_dataset_train", {}, "path_to_train_annotation_json", "path_to_train_images")
# %load_ext tensorboard

cfg = get_cfg()
cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
# cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml")
# cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
cfg.DATASETS.TRAIN = ("coco_2017_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 4  #batch size
#image resolution
cfg.INPUT.MIN_SIZE_TRAIN = (800,)  
cfg.INPUT.MAX_SIZE_TRAIN = 1333
# cfg.INPUT.MIN_SIZE_TRAIN = (300,)  
# cfg.INPUT.MAX_SIZE_TRAIN = 800

cfg.SOLVER.GRADIENT_ACCUMULATION = 2  # Accumulate gradients over 4 mini-batches
cfg.SOLVER.MAX_ITER = 40000  #Iteration
cfg.SOLVER.BASE_LR = 0.001  #Learning rate
cfg.SOLVER.GAMMA = 0.1  
# cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
# cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupStepWithFixedGammaLR"
# cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
cfg.SOLVER.STEPS = (30000, 350000)  # Adjust learning rate at these steps
cfg.SOLVER.NUM_DECAYS = len(cfg.SOLVER.STEPS)

cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.BASE_LR_END = 0.0
cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
# cfg.MODEL.PIXEL_STD = [57.375, 57.120, 58.395]
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
# cfg.SOLVER.STEPS = (210000, 250000)
# cfg.SOLVER.NUM_DECAYS = 3
# cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
# cfg.MODEL.ROI_HEADS.IN_FEATURES = ["res4"]
# cfg.MODEL.RPN.IN_FEATURES = ["res4"]
# cfg.MODEL.RPN.IN_FEATURES = ["res4"]
# cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
# cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64]]

# DIoU
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TRAIN = 0.5 
# cfg.MODEL.ROI_HEADS.NAME = "MyROIHeads"  #Trainer
cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "diou"
# cfg.MODEL.ROI_HEADS.NMS = cluster_nms
cfg.OUTPUT_DIR = "output"
# writer = SummaryWriter(log_dir="logs")
# trainer = DefaultTrainer(cfg, output_dir="output", tensorboard_writer=writer)
# print(input_shape.keys())


trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)

trainer.train()

evaluator = COCOEvaluator("coco_2017_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "coco_2017_val")
inference_on_dataset(trainer.model, val_loader, evaluator)

