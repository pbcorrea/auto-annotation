import json
import base64
import io
from PIL import Image

import numpy as np
from skimage.measure import find_contours
from skimage.measure import approximate_polygon

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES


CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
CONFIG_OPTS = ["MODEL.WEIGHTS", "model_final_971ab9.pkl", "MODEL.DEVICE", "cpu"]
CONFIDENCE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5


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
    context.logger.info("Run MaskRCNN - ResNet50")

    context.logger.info_with(
        'Got invoked',
		trigger_kind=event.trigger.kind,
		event_body=event.body
        )

    #1. Load data from request
    data = event.body

    context.logger.info(f"Receiving data ({type(data)}):  {data}")
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    image = convert_PIL_to_numpy(Image.open(buf), format="BGR")

    #2. Get predictions
    predictions = context.user_data.model_handler(image)
    instances = predictions['instances']

    #3. Cast predictions into CVAT format
    masks = instances.pred_masks
    scores = instances.scores
    labels = instances.pred_classes

    results = []
    for mask, score, label in zip(masks, scores, labels):
        label = COCO_CATEGORIES[int(label)]["name"]
        if score >= threshold:
            #4. Process masks
            mask = mask.numpy().astype(np.uint8)
            contours = find_contours(mask, MASK_THRESHOLD)
            contour = contours[0]
            contour = np.flip(contour, axis=1)
            contour = approximate_polygon(contour, tolerance=2.5)
            if len(contour) < 6:
                continue
            results.append({
                "confidence": str(score),
                "label": label,
                "points": contour.ravel().tolist(),
                "type": "polygon",
            })
    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)

