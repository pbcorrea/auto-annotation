from logging import getLogger
from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.data.detection_utils import read_image


logger = getLogger(__name__)


CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
CONFIG_OPTS = ["MODEL.WEIGHTS", "model_final_971ab9.pkl", "MODEL.DEVICE", "cpu"]
CONFIDENCE_THRESHOLD = 0.5

def setup_model_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
    cfg.merge_from_list(CONFIG_OPTS)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = CONFIDENCE_THRESHOLD
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_FILE)

    cfg.freeze()
    return cfg




if __name__ == "__main__":
    model_config = setup_model_config()
    image_path = "sample_frame.jpg"
    image = read_image(image_path, format="BGR")
    predictor = DefaultPredictor(model_config)
    predictions = predictor(image)
    instances = predictions['instances']
    pred_masks = instances.pred_masks
    scores = instances.scores
    pred_classes = instances.pred_classes
    for score, label in zip(scores, pred_classes):
        label = COCO_CATEGORIES[int(label)]["name"]
        print(float(score), label)
