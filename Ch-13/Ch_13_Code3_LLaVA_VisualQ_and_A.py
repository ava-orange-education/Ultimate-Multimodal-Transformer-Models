!pip install --upgrade -q accelerate bitsandbytes
!pip install git+https://github.com/huggingface/transformers.git
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
import torch
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
from google.colab import drive
drive.mount('/content/drive')
image_path="/path_to_folder/kids_play.jpg"
from PIL import Image
image=Image.open(image_path)
display(image)
prompt = "USER: <image>\nWhat do you see in the image? Describe in detail including color details\nASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
for k, v in inputs.items():
    print(k, v.shape)
# Generate
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    response = processor.decode(output[0], skip_special_tokens=True)
print(response)
