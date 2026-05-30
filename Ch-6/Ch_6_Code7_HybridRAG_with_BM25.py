!pip install llama_index
!pip install openai
!pip install llama-index-retrievers-bm25
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,Document,StorageContext
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
import os
from google.colab import drive
drive.mount('/content/drive')
api_key = input("Enter your OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = api_key
LLM = OpenAI(model="gpt-3.5-turbo")
documents=SimpleDirectoryReader('/path to folder/RAG_data_Gene_Protein').load_data()
combined_text = ""
for doc in documents:
    combined_text += doc.get_content() + "\n"
single_doc = Document(text=combined_text)
splitter = SentenceSplitter(chunk_size=500)
nodes = splitter.get_nodes_from_documents([single_doc])
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)
index = VectorStoreIndex(nodes, storage_context=storage_context)
# retireve the top 3 most similar nodes using embeddings
vector_retriever = index.as_retriever(similarity_top_k=3)
# retireve the top 3 most similar nodes using bm25
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3)

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()
    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)
        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes
hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
from llama_index.core.query_engine import RetrieverQueryEngine
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    llm=LLM,
)
response = query_engine.query(
    "What is the difference between FASTA and BLAST?")
from llama_index.core.response.notebook_utils import display_response
display_response(response)
