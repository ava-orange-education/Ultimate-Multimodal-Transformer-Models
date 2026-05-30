!pip3 install -q -U bitsandbytes==0.42.0
!pip3 install -q -U peft==0.8.2
!pip3 install -q -U accelerate==0.27.1
!pip3 install -q -U datasets==2.17.0
!pip3 install -q -U transformers==4.38.0
import os
import transformers
import torch
from google.colab import userdata
from datasets import load_dataset
from peft import LoraConfig,PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, GemmaTokenizer
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
model_id = "google/gemma-2b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config=bnb_config,
                                             device_map={"":0},
                                             token=os.environ['HF_TOKEN'])
text = "Quote: I was smiling yesterday,I am smiling today,"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
lora_config = LoraConfig(
    r = 8,
    target_modules = ["q_proj", "o_proj", "k_proj", "v_proj","gate_proj",  
    "up_proj", "down_proj"],
    task_type = "CAUSAL_LM",
)
from datasets import load_dataset
data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
def formatting_func(example):
    text = f"Quote: {example['quote'][0]}\nAuthor: {example['author'][0]}"
    return [text]
trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        num_train_epochs=100,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=-1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)
new_model = "/path_to_folder/Gemma_FineTuned" 
#Name of the model we will be pushing to huggingface model hub
# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
# Merge the model with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
merged_model= PeftModel.from_pretrained(base_model, new_model)
merged_model= merged_model.merge_and_unload()
text = "Quote: The most wasted of all days"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = merged_model.generate(**inputs, max_new_tokens=12)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
Quote: The most wasted of all days is one without laughter.
Author: Nicolas Chamfort
text = "Quote: Be yourself"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = merged_model.generate(**inputs, max_new_tokens=12)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
Quote: Be yourself; everyone else is already taken.
Author: Oscar Wilde
