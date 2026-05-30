!pip install llama_index openai
import openai
api_key = input("Enter your OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = api_key
openai.api_key = api_key
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
documents=SimpleDirectoryReader('/path_to_folder/RAG_data').load_data()
len(documents)
index=VectorStoreIndex.from_documents(documents,show_progress=True)
query_engine=index.as_query_engine()
response=query_engine.query("What is Craniosacral Therapy?")
from llama_index.core.response.pprint_utils import pprint_response pprint_response(response,show_source=True)
print(response)
