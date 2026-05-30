!pip install -q datasets
!pip install -q git+https://github.com/huggingface/transformers
!pip install -q bitsandbytes sentencepiece accelerate loralib
!pip install -q -U git+https://github.com/huggingface/peft.git
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
#This is the specific model class for IDEFICS. We will use this to load the pre-trained IDEFICS model
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
import torchvision.transforms as transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = "HuggingFaceM4/idefics-9b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", #NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=["lm_head", "embed_tokens"] 
)
processor = AutoProcessor.from_pretrained(checkpoint)
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, quantization_config=bnb_config, device_map="auto")
def do_inference(model, processor, prompts, max_new_tokens=50):
    tokenizer = processor.tokenizer
    bad_words = ["<image>", "<fake_token_around_image>"]
    bad_words_ids = None
    if len(bad_words) > 0:
        bad_words_ids = tokenizer(bad_words,  
    add_special_tokens=False).input_ids
    eos_token = "</s>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    inputs = processor(prompts, return_tensors="pt").to(device)
    generated_ids = model.generate(
        **inputs,
        eos_token_id=eos_token_id, 
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,   
    )
    generated_text = processor.batch_decode(generated_ids, 
    skip_special_tokens=True)[0]
    print(generated_text)
url = "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg"
prompts = [
    url,
    "Question: What's on the picture? Answer:",
]
do_inference(model, processor, prompts, max_new_tokens=50)
Question: What's on the picture? Answer: Two kittens playing in the grass.
def ds_transforms(example_batch):
  image_size = processor.image_processor.image_size
  image_mean = processor.image_processor.image_mean
  image_std = processor.image_processor.image_std
  image_transform = transforms.Compose([
      convert_to_rgb,
      #This is a common data augmentation technique. It crops a random portion of 
      #the image and then resizes it to the target size
      transforms.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0), 
      interpolation=transforms.InterpolationMode.BICUBIC),
      transforms.ToTensor(),
      transforms.Normalize(mean=image_mean, std=image_std)
  ])
  prompts = [] #Constructing Multimodal Prompts
  print(example_batch)
  for i in range(len(example_batch['caption'])):
    caption = example_batch['caption'][i].split(".")[0]
    #This creates a specific question-answering format that the model will learn 
    #to follow during fine-tuning. 
    prompts.append(
        [
            example_batch['image_url'][i],
            f"Question: What's on the picture? Answer: This is 
            {example_batch['name']}. {caption}",
        ],
    )
  inputs = processor(prompts, transform=image_transform,    
  return_tensors="pt").to(device)
  #For each image reference, it loads the image and applies the  
  #image_transform pipeline we defined
  inputs["labels"] = inputs["input_ids"]
  return inputs

ds = load_dataset("TheFusion21/PokemonCards")
ds = ds["train"].train_test_split(test_size=0.002) # Since the "PokemonCards" #dataset might only come with a "train" split, we need to create a separate #evaluation set to monitor the model's performance on unseen data during fine-#tuning.
train_ds = ds["train"]
eval_ds = ds["test"]
train_ds.set_transform(ds_transforms)
eval_ds.set_transform(ds_transforms)
model_name = checkpoint.split("/")[1]
config = LoraConfig(
    r = 16,
    lora_alpha = 32,
    target_modules = ["q_proj", "k_proj", "v_proj"],
    lora_dropout = 0.05,
    bias="none"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
trainable params: 19,750,912 || all params: 8,949,430,544 || trainable%: 0.2207
training_args = TrainingArguments(
    output_dir = f"{model_name}-PokemonCards",
    learning_rate = 2e-4,
    fp16 = True,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
    gradient_accumulation_steps = 8,
    dataloader_pin_memory = False,  
    save_total_limit = 3,
    eval_steps = 10,
    save_steps = 25,
    max_steps = 25,   #RUN for 25 optimization steps
    logging_steps = 5,
    remove_unused_columns = False,
    push_to_hub=False,
    label_names = ["labels"],
    load_best_model_at_end = False,
    report_to = "none",
    optim = "paged_adamw_8bit",
)
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_ds,
    eval_dataset = eval_ds
)
trainer.train()
url = "https://images.pokemontcg.io/sm9/63_hires.png"
from IPython.display import Image, display
display(Image(url=url))
prompts = [
    url,
    "Question: What's on the picture? Answer:",
]
do_inference(model, processor, prompts, max_new_tokens=100)
