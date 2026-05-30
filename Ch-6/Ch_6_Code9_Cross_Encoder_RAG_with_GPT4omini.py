!pip install langchain chromadb pypdf openai tiktoken sentence_transformers umap-learn
!pip install -U langchain-community
!pip install langchain_openai
import os
import time
import warnings
import numpy as np
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import pretty_print_result, pretty_print_docs, get_answer
from IPython.display import Markdown
from IPython.display import HTML, display
def set_css():
  display(HTML('''
  <style>
    pre {
        white-space: pre-wrap;
    }
  </style>
  '''))
get_ipython().events.register('pre_run_cell', set_css)
from google.colab import drive
drive.mount('/content/drive')
api_key = input("Enter your OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = api_key
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
loader = PyPDFLoader("/path_to_folder/Microsoft_2023.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, separators = ['\n\n','\n'])
splits = text_splitter.split_documents(data)
vectordb = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectordb.as_retriever(search_type="similarity",search_kwargs={"k": 7, "include_metadata": True})
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
query = "What is the revenue trend of Microsoft in 2022?"
answer, sources = get_answer(query, llm, retriever)
retrieved_documents = []
for doc in sources:
    retrieved_documents.append(doc.page_content)
pairs = [[query, doc] for doc in retrieved_documents]
scores = cross_encoder.predict(pairs)
sorted_indices = np.argsort(scores)[::-1]
final_df = pd.DataFrame()
for s, o in enumerate(sorted_indices):
    data = {'Document': [sources[o].page_content] , 'metadata': [str(sources[o].metadata)], 'Cross_encoder_Score':[scores[o]], 'Cross_encoder_Rank': [s+1]}
    df = pd.DataFrame(data)
    final_df = pd.concat([final_df,df],ignore_index=True)
# softmax function to convert cross-encoder scores to probabilities
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
final_df['relevance_score'] = softmax(final_df['Cross_encoder_Score'])
final_df = final_df[final_df['relevance_score'] >= 0.01]
from langchain.prompts import ChatPromptTemplate
from IPython.display import Markdown
qa_system_prompt = """
        You are a helpful AI assistant designed for answering questions based on the context provided. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        {context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "{question}"),
    ]
)
# filter for top documents based on cross encoder score
topn_docs = final_df[final_df['Cross_encoder_Rank']<=final_df.shape[0]]['Document'].tolist()
chain = qa_prompt | llm
result = chain.invoke({"question":query, "context": '\n\n'.join([doc for doc in topn_docs])})
Markdown(result.content.replace('$',"`$`"))
