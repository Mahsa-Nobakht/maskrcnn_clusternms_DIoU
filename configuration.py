import os

from detectron2.config import get_cfg
from detectron2 import model_zoo


def custom_config():
    cfg = get_cfg()
    file = open('data/coco.names', 'r')
    cfg.coco_name = [e.strip('\n') for e in file.readlines()]

    # get configuration from model_zoo
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    #
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml")


    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")

    #config defination
    # Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

    # Model
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cfg.coco_name)
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    # cfg.MODEL.BACKBONE.NAME = "build_retinanet_resnet_fpn_backbone"
    # cfg.MODEL.RESNETS.DEPTH = 34
    # cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
    # cfg.MODEL.RESNETS.RES5_DILATION = 1
    # cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0
    cfg.MODEL.BBOX_LOSS_TYPE = 'diou'
    # cfg.MODEL.BBOX_LOSS_TYPE = 'ciou'
    # cfg.MODEL.BBOX_LOSS_TYPE = 'our_diou_loss'
    # cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p3', 'p4', 'p5', 'p6']
    cfg.DATALOADER.NUM_WORKERS = 1
    # cfg.DATALOADER.NUM_WORKERS = 4


    # Solver
    cfg.SOLVER.BASE_LR = 0.001
    # cfg.SOLVER.BASE_LR = 0.002
    # cfg.SOLVER.BASE_LR = 0.005

    # cfg.SOLVER.MAX_ITER = 20780
    # cfg.SOLVER.MAX_ITER = 40000
    cfg.SOLVER.MAX_ITER = 40000
    # cfg.SOLVER.MAX_ITER = 16000
    # cfg.SOLVER.STEPS = (20, 500, 1000)
    cfg.SOLVER.STEPS = (30000,)
    # cfg.SOLVER.gamma = 0.5
    cfg.SOLVER.gamma = 0.1

    # options : WarmupMultiStepLR or WarmupCosaineLR
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"

    cfg.SOLVER.IMS_PER_BATCH = 2
    # cfg.SOLVER.IMS_PER_BATCH = 4
    # cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.MAX_PATIENCE = 10000
    cfg.VERBOSE = True

    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    # cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    # cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    # Test
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    # cfg.TEST.DETECTIONS_PER_IMAGE = 20


    # INPUT
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    # cfg.INPUT.FORMAT = "BGR"
    # cfg.INPUT.MASK_FORMAT = "polygon"  # alternative: "bitmask"

    #spesific test option
    # The period (in terms of steps) to evaluate the model during training.
    # Set to 0 to disable.
    # cfg..TEST.EVAL_PERIOD = 0

    # DATASETS
    # cfg.DATASETS.TEST = ('val',)
    cfg.DATASETS.TEST = ()
    cfg.DATASETS.TRAIN = ('train',)

    #RPN options
    # cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"  # used by RPN_HEAD_REGISTRY
    # Names of the input feature maps to be used by RPN
    # e.g., ["p2", "p3", "p4", "p5", "p6"] for FPN
    # cfg.MODEL.FPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    # cfg.MODEL.RPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    #ROI HEADS options
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # set threshold for this model
    # cfg.MODEL.WEIGHTS = os.path.join("/content/drive/My Drive/Capstone/output-79.0", "model_0006999.pth")
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    #Mask Head
    # Type of pooling operation applied to the incoming feature map for each RoI
    # cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE = "ROIAlignV2"

    #semantic segmentstion
    # cfg.MODEL.SEM_SEG_HEAD.NAME = "SemSegFPNHead"
    # cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]

    # DATASETS
    # cfg.OUTPUT_DIR = "output_standard_ourdiou"
    # cfg.OUTPUT_DIR = "output_standard_diou_itr7000_lr001"
    cfg.OUTPUT_DIR = "output_standard_diou_itr40000_lr001"
    # cfg.OUTPUT_DIR = "output_clusternms_ourdiou"
    # cfg.OUTPUT_DIR = "output_clusternms_ourdiou200"
    # cfg.OUTPUT_DIR = "output_clusternms_ourdiou5500_lr002"
    # cfg.OUTPUT_DIR = "output_clusternms_ourdiou_t6000_lr001_itr7000"
    # cfg.OUTPUT_DIR = "output_clusternms_diou_t7000_lr001_itr30000"
    # cfg.OUTPUT_DIR = "output_clusternms_diou_t7500_lr001_itr40000"
    # cfg.OUTPUT_DIR = "output_clusternms_diou_t7000_lr001_itr40000"
    # cfg.OUTPUT_DIR = "output_clusternms_ourdiou_t6000_lr001_itr40000"
    # cfg.OUTPUT_DIR = "output_clusternms_ourdiou_t6000_lr001_itr40000"
    # cfg.OUTPUT_DIR = "output_clusternms_ourdiou_t7000_lr001_itr40000"
    # cfg.OUTPUT_DIR = "output_clusternms_diou_t3000_lr001_itr40000"
    # cfg.OUTPUT_DIR = "output_clusternms_diou_t5000_lr001_itr7000"
    # cfg.OUTPUT_DIR = "output_cluster_stnd_diou2"
    # cfg.OUTPUT_DIR = "output_cluster_stnd_diou4000"
    # cfg.OUTPUT_DIR = "output_stnd_ciou"
    # cfg.OUTPUT_DIR = "output_cluster_ciou"
    # cfg.OUTPUT_DIR = "output_cluster_ciou5500"
    # cfg.OUTPUT_DIR = "output_standard_ourdiou_itr7000"







############Best_model####################
    # cfg.BEST_MODEL_DIR = "trained_model_diou/best_model"
    # cfg.BEST_MODEL_DIR = "trained_model_diou_itr7000_lr001/best_model"
    cfg.BEST_MODEL_DIR = "trained_model_diou_itr40000_lr001/best_model"
    # cfg.BEST_MODEL_DIR = "trained_model_ourdiou/best_model"
    # cfg.BEST_MODEL_DIR = "trained_model_ourdiou_itr7000/best_model"
    # cfg.BEST_MODEL_DIR = "cluster_stnd_diou/best_model"
    # cfg.BEST_MODEL_DIR = "cluster_stnd_diou4000/best_model"
    # cfg.BEST_MODEL_DIR = "trained_model_clusternms_ourdiou/best_model"
    # cfg.BEST_MODEL_DIR = "trained_model_cluster_our200/best_model"
    # cfg.BEST_MODEL_DIR = "trained_model_cluster_our_t6000_lr001_itr7000/best_model"
    # cfg.BEST_MODEL_DIR = "trained_model_cluster_diou_t7000_lr001_itr30000/best_model"
    # cfg.BEST_MODEL_DIR = "trained_model_cluster_diou_t7500_lr001_itr40000/best_model"
    # cfg.BEST_MODEL_DIR = "trained_model_cluster_diou_t7000_lr001_itr40000/best_model"
    # cfg.BEST_MODEL_DIR = "trained_model_cluster_ourdiou_t6000_lr001_itr40000/best_model"
    # cfg.BEST_MODEL_DIR = "trained_model_cluster_ourdiou_t7000_lr001_itr40000/best_model"
    # cfg.BEST_MODEL_DIR = "trained_model_cluster_diou_t3000_lr001_itr40000/best_model"
    # cfg.BEST_MODEL_DIR = "trained_model_stnd_ciou/best_model"
    # cfg.BEST_MODEL_DIR = "trained_model_cluster_ciou/best_model"




    cfg.SAVE_MODEL = True

    return cfg
