!pip install -q 'git+https://github.com/facebookresearch/segment-anything.git'
!pip install -q jupyter_bbox_widget roboflow dataclasses-json supervision==0.23.0
# download SAM weights
!mkdir -p {HOME}/weights
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P {HOME}/weights
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import supervision as sv
import numpy as np
import cv2
import torch
import os
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))
Next, we instantiate the device to leverage GPU acceleration when available.
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to (device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)
IMAGE_NAME = "dog_and_cat.jpg"
IMAGE_PATH = os.path.join(HOME,IMAGE_NAME)
Now we read the image with OpenCV (BGR format), convert to RGB, then feed directly to mask_generator for automatic zero-shot segmentation, which outputs a list of dictionary with masks, scores, and bounding boxes.
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
sam_result = mask_generator.generate(image_rgb)
print(sam_result[0].keys())
dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'])
Using Supervision's MaskAnnotator, let us convert SAM masks to Detections, overlay colorful segments on the test image.
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
detections = sv.Detections.from_sam(sam_result=sam_result)
annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
sv.plot_images_grid(
    images=[image_bgr, annotated_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)
masks = [
    mask['segmentation']
    for mask
    in sorted(sam_result, key=lambda x: x['area'], reverse=True)
]
len(masks)
23
top_masks = masks[:4]
sv.plot_images_grid(
    images=top_masks,
    grid_size=(2, 2),  
    size=(16, 16)
)
mask_predictor = SamPredictor(sam)
# helper function that loads an image before adding it to the widget
import base64
def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/jpg;base64,"+encoded
IS_COLAB = True
if IS_COLAB:
    from google.colab import output
    output.enable_custom_widget_manager()
from jupyter_bbox_widget import BBoxWidget
widget = BBoxWidget()
widget.image = encode_image(IMAGE_PATH)
widget
roi_box = widget.bboxes
box = widget.bboxes[0] if widget.bboxes else roi_box
box = np.array([
    box['x'],
    box['y'],
    box['x'] + box['width'],
    box['y'] + box['height']
])
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
mask_predictor.set_image(image_rgb)
masks, scores, logits = mask_predictor.predict(
    box=box,
    multimask_output=True
)
#color will be selected based on the index of each detection.
box_annotator = sv.BoxAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX) 
mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)
#Packs both masks and their bounding boxes into a Detections object.
detections = sv.Detections(
    xyxy=sv.mask_to_xyxy(masks=masks),
    mask=masks
)
detections = detections[detections.area == np.max(detections.area)] #Filters the detections to only keep the one with the largest area.
source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
#Draws the segmentation mask (usually as a filled colored overlay) on a copy of the original image.
segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
sv.plot_images_grid(
    images=[source_image, segmented_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)
