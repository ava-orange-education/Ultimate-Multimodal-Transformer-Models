# Install dependencies
!pip install transformers==4.41.2
!pip install torch torchvision torchaudio --index-url  https://download.pytorch.org/whl/cu121 -q
!pip install pillow requests accelerate bitsandbytes -q
!pip install openai -q  # For GPT-4V
# Key imports
import torch
from PIL import Image
import numpy as np
import requests
import cv2
from transformers import AutoProcessor, AutoModelForVision2Seq
from google.colab import drive
drive.mount('/content/drive')  # Optional for custom images
model_id = "microsoft/kosmos-2-patch14-224"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True)
model = model.to(torch.bfloat16).cuda().eval() if torch.cuda.is_available() else model.eval()
# Load image
url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/annotated_snowman.jpg"
image = Image.open(requests.get(url, stream=True).raw)
# Grounded query for object detection/grounding
prompt = "<grounding> An image of"
inputs = processor(text=prompt, images=image, return_tensors="pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
with torch.no_grad():
    inputs = {k: v.to(device, dtype=torch.bfloat16) if k in    
             ['pixel_values'] else v.to(device)for k, v in   
             inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_text = processor.batch_decode(generated_ids, 
    skip_special_tokens=True)[0]
processed_text, entities = processor.post_process_generation(generated_text)
print("Grounded Description:", processed_text)
print("Entities with BBoxes:", entities)
def draw_boxes_safe(image, entities):
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    # Handle different entities formats from Kosmos-2
    if isinstance(entities, dict):
        for entity_name, bbox_data in entities.items():
            if isinstance(bbox_data, (list, tuple)) and len(bbox_data) > 0:
                bbox = bbox_data[0]  # First bbox instance
                x1, y1, x2, y2 = [int(coord * min(h, w)) for coord in bbox]
                cv2.rectangle(img_array, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(img_array, entity_name, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    elif isinstance(entities, list):
        for item in entities:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                entity = item[0] if isinstance(item[0], str) else str(item[0])
                bbox = item[1] if len(item) > 1 else None
                if bbox and len(bbox) >= 4:
                    x1, y1, x2, y2 = [int(coord * min(h, w)) for coord in bbox]
                    cv2.rectangle(img_array, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(img_array, entity[:20], (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return Image.fromarray(img_array)
try:
    grounded_img = draw_boxes_safe(image, entities)
    grounded_img  # Displays in Colab
except Exception as e:
    print(f"Visualization failed: {e}")
    print("Raw entities for debugging:", entities)
grounded_img
import openai
openai.api_key = "your-openai-key"
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": [{"type": "text", "text": "Describe this image in detail, including any text."}, {"type": "image_url", "image_url": {"url": url}}]}]
)
print(response.choices[0].message.content)
