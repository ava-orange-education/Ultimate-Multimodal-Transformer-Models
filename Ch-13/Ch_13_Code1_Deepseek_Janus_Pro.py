%cd Janus
!pip install -e .
!pip install flash-attn
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import textwrap
!git clone https://github.com/deepseek-ai/Janus.gitmodel_path="deepseek-ai/Janus-Pro-1B"
vl_chat_processor=VLChatProcessor.from_pretrained(model_path)
tokenizer=vl_chat_processor.tokenizer
vl_gpt=AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True)
vl_gpt=vl_gpt.to(torch.bfloat16).cuda().eval()
from google.colab import drive
drive.mount('/content/drive')
image_path="/path_to_folder/DL/person_living_room.jpg"
question="Can you identify objects in the image and list based on the category/type"
conversation=[{"role":"<|User|>","content":f"<image_placeholder>\n{question}","images":[image_path]},{"role":"<|Assistant|>","content":""}] #The string combines the image placeholder with the question text.
#Assistant's response and is initially empty. The model will fill this with its generated response.
#load image
pil_images=load_pil_images(conversation)
#prepare inputs for the model
prepare_inputs=vl_chat_processor(conversations=conversation,images=pil_images,force_batchify=True).to(vl_gpt.device) #ensures that the inputs are batched (even if it's a single sample).
inputs_embeds=vl_gpt.prepare_inputs_embeds(**prepare_inputs) 
outputs=vl_gpt.language_model.generate(
inputs_embeds=inputs_embeds, #embedded multimodal input.
attention_mask=prepare_inputs.attention_mask, #tells the model which tokens to attend to (and which to ignore).
pad_token_id=tokenizer.eos_token_id, #special token for padding
bos_token_id=tokenizer.bos_token_id, #special token for beginning of sequence
eos_token_id=tokenizer.eos_token_id, #special token for end of sequence
max_new_tokens=512, #limits how many tokens the model can generate.
                                       do_sample=False, #deterministic
                                       use_cache=True) #enables caching for speed during decoding.
# Decode and print response
#If the output tensor is on the GPU, this moves it to the CPU so it can be converted into a list (since .tolist() only works on CPU tensors).
answer=tokenizer.decode(outputs[0].cpu().tolist(),skip_special_tokens=True) #Translates the list of token IDs back into a human-readable string.
# Format and print the summary as a paragraph
wrapped_para = textwrap.fill(answer, width=125)
print(f"{prepare_inputs['sft_format'][0]}",wrapped_para)
question="Can you write an introduction scene for a novel based on this picture, assume the character name is Sofia, who is a private detective with her dog Micky" 
conversation=[{"role":"<|User|>","content":f"<image_placeholder>\n{question}","images":[image_path]},{"role":"<|Assistant|>","content":""}]
answer=tokenizer.decode(outputs[0].cpu().tolist(),skip_special_tokens=True)
wrapped_para = textwrap.fill(answer, width=125)
print(f"{prepare_inputs['sft_format'][0]}",wrapped_para)
