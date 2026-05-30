!pip install transformers datasets torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,GenerationConfig
hugging_face_dataset_name="gopalkalpande/bbc-news-summary"
dataset=load_dataset(hugging_face_dataset_name)
import textwrap
example_indices=[52,45]
dash_line='-'.join('' for x in range(100))
for i,index in enumerate(example_indices):
    print(dash_line)
    print('Example',i+1)
    print(dash_line)
    print('News:')
    print(textwrap.fill(dataset['train'][index]['Articles'],width=100))
    print(dash_line)
    print('Baseline_Human_Headline:')
    print(textwrap.fill(dataset['train'][index]['Summaries'],width=100))
    print(dash_line)
    print()
model_name='google/flan-t5-base'
model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name,use_fast=True)
#zero shot inference
for i,index in enumerate(example_indices):
    news=dataset['train'][index]['Articles']
    summary=dataset['train'][index]['Summaries']
    prompt= f"""
Summarize the following news article.
{news}
Summary:
""" 
    inputs=tokenizer(prompt,return_tensors='pt')
    output=tokenizer.decode(model.generate(inputs["input_ids"],max_new_tokens=50)[  
    0],skip_special_tokens=True)
    print(dash_line)
    print('Example ',i+1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)
    print(f'MODEL GENERATED SUMMARY WITH ZERO SHOT:\n{output}\n')
def make_prompt(example_indices_full,example_index_to_summarize):
  prompt = ''
  for index in example_indices_full:
    news=dataset['train'][index]['Articles']
    summary=dataset['train'][index]['Summaries']
    # the stop sequence '{summary}\n\n\n' is important for FLAN-T5, other models    
    may have their own preferred stop sequences
    prompt += f"""
News:
{news}
summary of above news article
{summary}
"""
  news=dataset['train'][example_index_to_summarize]['Articles']
  prompt += f"""
News:
{news}
summarize the above news article?
"""
  return prompt
example_indices_full = [40]
example_index_to_summarize = 52
one_shot_prompt=make_prompt(example_indices_full,example_index_to_summarize)
summary=dataset['train'][example_index_to_summarize]['Summaries']
inputs=tokenizer(one_shot_prompt,return_tensors='pt')
output=tokenizer.decode(model.generate(inputs["input_ids"],max_new_tokens=50)[0],skip_special_tokens=True)
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATED SUMMARY WITH ONE SHOT:\n{output}')
example_indices_full = [40,50]
example_index_to_summarize = 52
few_shot_prompt=make_prompt(example_indices_full,example_index_to_summarize)
summary=dataset['train'][example_index_to_summarize]['Summaries']
inputs=tokenizer(few_shot_prompt,return_tensors='pt')
output=tokenizer.decode(model.generate(inputs["input_ids"],max_new_tokens=35)[0],skip_special_tokens=True)
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATED SUMMARY WITH FEW SHOT:\n{output}')
