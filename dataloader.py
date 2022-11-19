import json
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2.data.datasets.coco as dscoco


def load_data(t="train"):
    if t == "train":
        with open("data/annotations/instances_train2017.json", 'r') as file:
            train = json.load(file)

        for i in train['images']:
            i['file_name'] = 'data/coco/images/train2017/' + i['file_name']

        train_dict = dscoco.load_coco_json('data/annotations/instances_train2017.json', 'data/coco/images/train2017')
        return train_dict

    elif t == "val":
        with open("data/annotations/instances_val2017.json", 'r') as file:
            val = json.load(file)

        for i in val['images']:
            i['file_name'] = 'data/coco/images/val2017/' + i['file_name']

    val_dict = dscoco.load_coco_json('data/annotations/instances_val2017.json', 'data/coco/images/val2017')
    return val_dict

# for d in ["train", "val"]:
#     DatasetCatalog.register(d, lambda d=d: load_data(d))
#     MetadataCatalog.get(d).set(thing_classes=["Dog", "Cat", "Mouse"])
# metadata = MetadataCatalog.get("train")
