!curl https://ollama.ai/install.sh | sh 
!nohup ollama serve & 
!ollama pull deepseek-r1:14b 
!pip install ollama
import ollama
response = ollama.chat(model='deepseek-r1:14b', messages=[
  {
    'role': 'user',
    'content': "What is Knowledge Distillation in Large Language Models?",
  },
])
print(response['message']['content'])
