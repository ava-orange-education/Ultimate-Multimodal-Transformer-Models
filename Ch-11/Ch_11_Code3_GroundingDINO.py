!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
!pip install supervision
!pip install -q roboflow
!pip install -q -e .
import os,torch
import supervision as sv
from PIL import Image
import cv2
from groundingdino.util.inference import load_model, load_image, predict, annotate
%matplotlib inline
%cd {HOME}
!git clone https://github.com/IDEA-Research/GroundingDINO.git
%cd GroundingDINO/
CONFIG_PATH = os.path.join(HOME, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))
%cd {HOME}
!mkdir {HOME}/weights
%cd {HOME}/weights
!wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))
/content/GroundingDINO/weights/groundingdino_swint_ogc.pth ; exist: True
%cd {HOME}
detection_model = load_model(CONFIG_PATH, WEIGHTS_PATH)
def inference_pipeline(image_path, text_prompt, box_threshold=0.3, text_threshold=0.3):
image_source, image = load_image(image_path)
boxes, logits, phrases = predict(
model=detection_model, 
image=image, 
caption=text_prompt, 
box_threshold=box_threshold, 
text_threshold=text_threshold
)
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
sv.plot_image(annotated_frame, (16, 16))
!wget https://prtimes.jp/i/1355/5420/resize/d1355-5420-149832-0.jpg -O /path_to_folder/soccer.jpg
inference_pipeline("/path_to_folder/soccer.jpg", "player,ball", 0.2, 0.2)
