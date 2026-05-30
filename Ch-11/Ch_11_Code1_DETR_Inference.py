import torch
import requests
from PIL import Image,ImageDraw,ImageFont
import torchvision.transforms as T
model = torch.hub.load('facebookresearch/detr','detr_resnet101',pretrained=True)
model.eval()
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(dev)
from google.colab import drive
drive.mount("/content/drive")
img=Image.open('/content/drive/MyDrive/DL/dog_and_cat.jpg').convert('RGB').resize((400, 600))
img
#normalize the input image suitable for ResNet
transform=T.Compose([
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
img_tensor=transform(img).unsqueeze(0)
img_tensor.shape
torch.Size([1, 3, 600, 400])
img_tensor = img_tensor.to(dev)
for logits,boxes in zip(output['pred_logits'][0],output['pred_boxes'][0]):
  cls=logits.argmax()
  if cls>=len(CLASSES):  #to skip the null object
    continue
  label=CLASSES[cls]
  print(label)
dog
cat
# Convert image to torch tensor for visualization (uint8, [0,255])
from torchvision.transforms.functional import to_tensor
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert  
# Ensure image is in the correct format
img_tensor_viz = to_tensor(img).mul_(255).byte()
# Convert boxes from center_cxcywh to xyxy
# We assume 'boxes' are currently [cx, cy, w, h]
boxes_converted = box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
# Draw boxes using the converted coordinates
result = draw_bounding_boxes(
    img_tensor_viz, 
    boxes_converted.long(), 
    labels=[f"{label} {score:.0%}" for label, score in zip(labels, output['pred_logits'].softmax(-1)[0, keep, :-1].max(-1)[0])], 
    colors="red", 
    width=4, 
    font_size=18
)
result_img = result.permute(1,2,0).numpy()
# Display
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8)); plt.imshow(result_img); plt.axis('off'); plt.show()
