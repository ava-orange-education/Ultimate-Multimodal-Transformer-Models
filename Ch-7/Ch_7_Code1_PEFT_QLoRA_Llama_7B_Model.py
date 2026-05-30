!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.0 transformers==4.31.0 trl==0.4.7
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
<s>[INST] <<SYS>> 
System prompt <</SYS>> 
User prompt [/INST] Model answer </s>. 
!pip install -q datasets
!huggingface-cli login
from datasets import load_dataset
import re
# Load the dataset
dataset = load_dataset('timdettmers/openassistant-guanaco')
# Shuffle the dataset and slice it
dataset = dataset['train'].shuffle(seed=42).select(range(1000))
# Define a function to transform the data
def transform_conversation(example):
    conversation_text = example['text']
    segments = conversation_text.split('###')
    reformatted_segments = []
    # Iterate over pairs of segments
    for i in range(1, len(segments) - 1, 2):
        human_text = segments[i].strip().replace('Human:', '').strip()
        # Check if there is a corresponding assistant segment before processing
        if i + 1 < len(segments):
            assistant_text = segments[i+1].strip().replace('Assistant:', '').strip()
            # Apply the new template
            reformatted_segments.append(f'<s>[INST] {human_text} [/INST] {assistant_text} </s>')
        else:
            # Handle the case where there is no corresponding assistant segment
            reformatted_segments.append(f'<s>[INST] {human_text} [/INST] </s>')
    return {'text': ''.join(reformatted_segments)}
# Apply the transformation
transformed_dataset = dataset.map(transform_conversation)
transformed_dataset.push_to_hub("guanaco-llama2-1k")
from google.colab import drive
drive.mount('/content/drive')
# Set the name for Fine-tuned model
new_model = "/path_to_folder/Llama-2-7b-chat-finetune"
# Enable gradient checkpointing
gradient_checkpointing = True
dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
# Indicates that the model should be loaded with 4-bit quantization.
    bnb_4bit_quant_type="nf4", 
#Specifies the type of 4-bit quantization to use
    bnb_4bit_compute_dtype="float16", 
#Specifies that computations should be done in float16 precision.
    bnb_4bit_use_double_quant=False, 
#Double quantization typically means applying quantization in two steps for potentially better accuracy but is not enabled here.
) 
if compute_dtype == torch.float16 and True:
    major, _ = torch.cuda.get_device_capability() #This line retrieves the major and minor CUDA capability of the currently selected GPU.
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)
# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1  #setting tensor parallelism
# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf", trust_remote_code=True)
#This parameter allows the code to trust the remote repository's code
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
#Proper padding can help maintain numerical stability and avoid issues that might arise from incorrectly processed sequences.
# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
# Set training parameters
training_arguments = TrainingArguments(
    output_dir="/path_to_folder/LORA", #Specifies the directory where the trained  
    model and other outputs (like checkpoints and logs) will be saved.
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1, #gradients are updated every step.
    optim="paged_adamw_32bit", #optimizer configured to work with 32-bit 
    precision(The optimizer only keeps the necessary pages (subsets of parameters) 
    in memory while performing updates, reducing the overall memory footprint.)
    save_steps=0, #checkpoints will not be saved during training
    logging_steps=25, #means metrics are logged every 25 batches
    learning_rate=2e-4,
    weight_decay=0.001, #Specifies the weight decay (L2 regularization)     
    coefficient. This helps prevent overfitting by penalizing large weights
    fp16=True,
    bf16=False,
    max_grad_norm=2e-4, # Clipping helps to prevent exploding gradients by scaling 
    gradients that exceed this threshold
    max_steps=-1, #Setting this to -1 means there is no limit, and training will 
    run for the full number of epochs
    warmup_ratio=0.03, #Warmup gradually increases the learning rate from 0 to the 
    initial learning rate over this fraction of steps.
    group_by_length=True, #ensuring more uniform sequence lengths within a batch
    lr_scheduler_type="cosine", #A "cosine" scheduler adjusts the learning rate 
    following a cosine curve, which can help improve training dynamics.
    report_to="tensorboard" #to visualize and track training progress.
)
# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=None, #It will use the model's default maximum sequence length.
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False, #Indicates whether to use packing (concatenating multiple sequences into one) during training. Setting it to False means no packing is used.
)
# Train model
trainer.train()
# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf",
    low_cpu_mem_usage=True, #Optimizes memory usage on the CPU
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
model = PeftModel.from_pretrained(base_model, new_model) 
# The directory where the fine-tuned model with LoRA weights is saved
model = model.merge_and_unload()
#Merges the LoRA weights with the base model's weights and unloads the additional overhead.
#This results in a single model that combines both the base model's weights and the fine-tuned LoRA weights, optimizing for inference or further training.
# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# suppresses all warnings and informational messages, only displaying critical errors.
logging.set_verbosity(logging.CRITICAL)
prompt="Can you create a function in Javascript that returns a string of the current date in this format? February 6, 2023"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)
#The special tokens <s>[INST] and [/INST] are added to the prompt, which might be specific to the
#model's expected input format for instruction-based or conversational contexts.
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
<s>[INST] Can you create a function in Javascript that returns a string of the current date in this format? February 6, 2023 [/INST] Sure! Here is a function in JavaScript that returns a string of the current date in the format "February 6, 2023":
function getCurrentDate() {
  const date = new Date();
  const month = date.getMonth() + 1;
  const day = date.getDate();
  const year = date.getFullYear();
  return `${month} ${day}, ${year}`;
}
You can call this function by calling `getCurrentDate()` and it will return the current date in the format "February 6, 2023".
I hope this helps! Let me know if you have any questions.
