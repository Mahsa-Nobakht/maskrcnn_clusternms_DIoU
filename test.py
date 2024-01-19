from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
# from detectron2.config import get_cfg
import os
from detectron2.utils.visualizer import ColorMode
import configuration as cfg
import json

def get_board_dicts(imgdir):
    json_file = imgdir+"/instances_minival2014.json"
    with open(json_file) as f:
        dataset_dicts = json.load(f)
    for i in dataset_dicts:
        # f = i["info"]
        filename = i["file_name"]
        i["file_name"] = imgdir+"/"+filename
        for j in i["annotations"]:
            j["bbox_mode"] = BoxMode.XYWH_ABS
            label = j["category_id"]
            if label == "##::OTHER":
              label = 2
            else:
              j["category_id"] = int(label)
    return dataset_dicts
file = open('data/coco.names', 'r')
coco_name = [e.strip('\n') for e in file.readlines()]

for d in ["train", "val"]:
    DatasetCatalog.register("boardetect_" + d, lambda d=d: get_board_dicts("board1/" + d))
    MetadataCatalog.get("boardetect_" + d).set(thing_classes=coco_name)
board_metadata = MetadataCatalog.get("boardetect_val")

dataset_dicts = get_board_dicts("data/annotations")
print(dataset_dicts)
for d in random.sample(dataset_dicts, 10):
    img = cv2.imread(d["file_name"])
    print(d)
    visualizer = Visualizer(img[:, :, ::-1], metadata=board_metadata)
    vis = visualizer.draw_dataset_dict(d)
    cv2_imshow(vis.get_image()[:, :, ::-1])
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set threshold for this model
# cfg.MODEL.WEIGHTS = os.path.join("/content/drive/My Drive/Capstone/output-79.0", "model_0006999.pth")
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
predictor = DefaultPredictor(cfg)

dataset_dicts = get_board_dicts("board1/train")
for d in random.sample(dataset_dicts, 8):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=board_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(v.get_image()[:, :, ::-1])