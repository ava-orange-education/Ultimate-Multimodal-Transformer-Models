!pip install ftfy regex tqdm
!pip install git+https://github.com/openai/CLIP.git
!pip install scikit-image
skimage.data.download_all(directory=None)
import torch
print("Torch version:", torch.__version__)
Torch version: 2.10.0+cu128
from collections import OrderedDict
from pkg_resources import packaging
import matplotlib.pyplot as plt
import IPython.display
from PIL import Image
import numpy as np
import skimage
import torch
import clip
import os
%matplotlib inline
clip.available_models()
model, preprocess = clip.load("RN50x64")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)
Model parameters: 623,258,305
Input resolution: 448
Context length: 77
Vocab size: 49408
descriptions = {
    "rocket": "a rocket standing on a launchpad",
}
original_images = []
images = []
texts = []
plt.figure(figsize=(16, 5))
for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
    name = os.path.splitext(filename)[0]
    if name not in descriptions:
        continue
    image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")
    plt.subplot(2, 4, len(images) + 1)
    plt.imshow(image)
    plt.title(f"{filename}\n{descriptions[name]}")
    plt.xticks([])
    plt.yticks([])
    original_images.append(image)
    images.append(preprocess(image))
    texts.append(descriptions[name])
plt.tight_layout()
descriptions = [
    "A photo of a sky scrapper ",
    "A photo of pillars ",
    "A photo of a Rocket",
    ]
text_tokens = clip.tokenize(descriptions).cuda()
with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True) 
top_probs, top_labels = text_probs.cpu().topk(3, dim=-1)
plt.figure(figsize=(16, 16))
for i, image in enumerate(original_images):
    plt.subplot(1,2, 2 * i + 1)
    plt.imshow(image)
    plt.axis("off")
    plt.subplot(1, 2, 2 * i + 2)
    y = np.arange(top_probs.shape[-1])
    plt.grid()
    plt.barh(y, top_probs[i])
    plt.gca().invert_yaxis()
    plt.gca().set_axisbelow(True)
    plt.yticks(y, [descriptions[index] for index in  
    top_labels[i].numpy()])
    plt.xlabel("probability")
plt.subplots_adjust(wspace=0.5)
plt.show()
