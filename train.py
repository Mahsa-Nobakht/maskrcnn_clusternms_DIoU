<<<<<<< HEAD
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

=======
import os

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from detectron2.engine import DefaultTrainer, DefaultPredictor
# import configuration as cfg
from configuration import custom_config
from dataloader import load_data
import torch

from mask_rcnn_prj.best_model import BestCheckpointer
from mask_rcnn_prj.utils.validation_loss import ValidationLoss

torch.cuda.empty_cache()

import fvcore.nn

if __name__ == '__main__':
    cfg = custom_config()

    for d in ["train", "val"]:
        DatasetCatalog.register(d, lambda d=d: load_data(d))
        MetadataCatalog.get(d).set(thing_classes=cfg.coco_name)
    metadata = MetadataCatalog.get("train")


    # cfg = cfg.custom_config(len(cfg.coco_name))
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.BEST_MODEL_DIR, "early_stopping_model0.pth")
    #
    # # cfg.DATASETS.TEST = ("val",)
    # # predictor = DefaultPredictor(cfg)
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # trainer.train()

    ######## Generate validation Loss during training #######
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    cfg.DATASETS.TEST = ("val",)
    val_loss = ValidationLoss(cfg)  ## Additional parameters
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=True)
    trainer.train()

    ######## Evaluate train model ############
    #
    # import the COCO Evaluator to use the COCO Metrics
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    from detectron2 import model_zoo

    cfg.DATASETS.TEST = ("val",)
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0006999.pth")  # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

    # cfg.MODEL.WEIGHTS = os.path.join(cfg.BEST_MODEL_DIR, "early_stopping_model.pth")  # path to the model we just trained


    predictor = DefaultPredictor(cfg)
    # Call the COCO Evaluator function and pass the Validation Dataset
    evaluator = COCOEvaluator("val", cfg, False, output_dir="/output/")
    val_loader = build_detection_test_loader(cfg, "val")

    # Use the created predicted model in the previous step
    inference_on_dataset(predictor.model, val_loader, evaluator)

########################
    # evaluator = COCOEvaluator("train", cfg, False, output_dir="/output/")
    # train_loader = build_detection_test_loader(cfg, "train")
    #
    # # Use the created predicted model in the previous step
    # inference_on_dataset(trainer.model, train_loader, evaluator)

###########################
    # evaluator = COCOEvaluator("val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    # # evaluator = COCOEvaluator("val", cfg, False, output_dir=cfg.BEST_MODEL_DIR)
    # val_loader = build_detection_test_loader(cfg, "val")
    # result = inference_on_dataset(trainer.model, val_loader, evaluator)
    ######################################
    #
    # from detectron2.engine import DefaultTrainer
    # from detectron2 import model_zoo
    #
    # # cfg = get_cfg()
    # # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml"))
    # # # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml")
    # # cfg.DATASETS.TRAIN = ("train",)
    # # cfg.DATASETS.TEST = ()
    # # cfg.DATALOADER.NUM_WORKERS = 2
    # # # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    # # #     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml")
    # #
    # # cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    # # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # # cfg.SOLVER.MAX_ITER = 10  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    # # cfg.SOLVER.STEPS = []  # do not decay learning rate
    # # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    # # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # # # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    # # cfg.CUDA_LAUNCH_BLOCKING=1
    #
    # # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # # trainer = DefaultTrainer(cfg)
    # # trainer.resume_or_load(resume=False)
    # # trainer.train()
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg)
    # # cfg.DATASETS.TEST = ("val",)
    # cfg.DATASETS.VAL = ("val",)
    # val_loss = ValidationLoss(cfg)  ## Additional parameters
    # trainer.register_hooks([val_loss])
    # # swap the order of PeriodicWriter and ValidationLoss
    # trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    # # trainer.register_hooks([BestCheckpointer()])
    #
    # trainer.resume_or_load(resume=True)
    # trainer.train()
    #
    # # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "final_model.pth")  # path to the model we just trained
    # # final = os.path.join(cfg.OUTPUT_DIR, "final_model.pth")  # path to the model we just trained
    # # best = os.path.join(cfg.BEST_MODEL_DIR, "early_stopping_model0.pth")  # path to the model we just trained
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set a custom testing threshold
    # # predictor = DefaultPredictor(cfg)
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cfg.coco_name)
    # cfg.DATASETS.TEST = ("val",)
    #
    # from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    # from detectron2.data import build_detection_test_loader
    #
    # evaluator = COCOEvaluator("val", output_dir="./output")
    # val_loader = build_detection_test_loader(cfg, "val")
    # inference_on_dataset(trainer.model, val_loader, evaluator)
>>>>>>> origin/main
