import json
import base64
import io
from PIL import Image

import numpy as np
import pycocotools.mask as mask_util
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.structures import BoxMode


CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
CONFIG_OPTS = ["MODEL.WEIGHTS", "model_final_971ab9.pkl", "MODEL.DEVICE", "cpu"]
CONFIDENCE_THRESHOLD = 0.5

def init_context(context):
    context.logger.info(f"Initializing model context")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
    cfg.merge_from_list(CONFIG_OPTS)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = CONFIDENCE_THRESHOLD
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_FILE)
    cfg.freeze()

    model = DefaultPredictor(cfg)
    context.user_data.model_handler = model

    context.logger.info(f"Model initialized successfully!")



def handler(context, event):
    context.logger.info(f"Receiving label request")

    #1. Load data from request
    data = event.body

    context.logger.info(f"Receiving data ({type(data)}):  {data}")
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image_id = int(data.get("image_id", -1))
    threshold = float(data.get("threshold", 0.5))
    image = convert_PIL_to_numpy(Image.open(buf), format="BGR")

    #2. Get predictions
    predictions = context.user_data.model_handler(image)
    instances = predictions['instances']
    
    #3. Cast predictions into
    bboxes = BoxMode.convert(instances.pred_boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    scores = instances.scores
    labels = instances.pred_classes
    rles = [mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0] for mask in instances.pred_masks]
    for rle in rles:
        rle["counts"] = rle["counts"].decode("utf-8")

    results = []

    #4. Filter and retrieve results
    for annotation_idx, (score, mask, bbox, label) in enumerate(zip(scores, rles, bboxes, labels), 1):
        label = COCO_CATEGORIES[int(label)]["name"]
        if score >= threshold:
            results.append({
                "id": annotation_idx,
                "image_id": image_id,
                "category_id": int(label),
                "segmentation": mask,
                "bbox": bbox,
                "iscrowd": 0,
            })
    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)
