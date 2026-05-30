!pip install -U llama-index llama-index-llms-gemini llama-index-embeddings-gemini
!pip install –U llama-index-vector-stores-pinecone pinecone-client
import os
gemini_api_key = input("Enter your Google Gemini API key: ")
pinecone_api_key = input("Enter your Pinecone API key: ")
pinecone_env = input("Enter your Pinecone environ name (e.g. us-east-1-aws): ")
os.environ["GEMINI_API_KEY"] = gemini_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ["PINECONE_ENVIRONMENT"] = pinecone_env
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
# Gemini chat model (analogous to ChatOpenAI with gpt-4o-mini)
llm = Gemini(model_name="models/gemini-1.5-pro",temperature=0.0)
# Gemini embedding model (you can keep HuggingFace if you prefer)
embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001")
# Global settings, analogous to our previous section
Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)
Settings.num_output = 2000
Settings.context_window = 4096
import pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
# Initialize Pinecone client
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT"],
)
index_name = "medical-case-rag-index"
# Create or connect to Pinecone index
if index_name not in pinecone.list_indexes():
   pinecone.create_index(name=index_name, dimension=768, metric="cosine")
pc_index = pinecone.Index(index_name)
# Wrap Pinecone index for LlamaIndex
vector_store = PineconeVectorStore(pinecone_index=pc_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
)
from llama_index.core.response.pprint_utils import pprint_response
query_engine = index.as_query_engine()
response = query_engine.query(
    "What is SLE assessment? Explain with a use case and draft an email in Hindi."
)
pprint_response(response, show_source=True)
print(response)
