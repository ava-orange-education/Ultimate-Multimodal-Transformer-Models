!pip install diffusers==0.11.1
!pip install transformers scipy ftfy accelerate
!pip install --upgrade huggingface-hub==0.26.2 transformers==4.46.1 tokenizers==0.20.1 diffusers==0.31.0
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
if torch.cuda.is_available():
    device=torch.device("cuda")
elif torch.backends.mps.is_available():
    device=torch.device("mps")
pipe = pipe.to(device)
prompt = "a low resolution 100x100 photograph of a sea shore during a sunset"
image = pipe(prompt).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)
image
generator = torch.Generator(device).manual_seed(1024)
image = pipe(prompt, num_inference_steps=50, generator=generator).images[0]
image
from PIL import Image
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
num_images = 3
prompt = ["a photograph of a cat riding a car"] * num_images
images = pipe(prompt).images
grid = image_grid(images, rows=1, cols=3)
grid
!pip3 install numpy --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
!pip3 install torch torchvision torchaudio
!pip install transformers scipy ftfy accelerate ipywidgets
!pip install diffusers
import torch
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
    , torch_dtype=torch.float16
).to("cuda:0")
# Generate an image
prompt = "a color photograph of a cat driving in a car"
image = pipe(
    prompt, num_inference_steps=30
).images[0]
image
prompt = "a photograph of a cyclist in a busy road"
image = pipe(
    prompt, num_inference_steps=30
).images[0]
image
