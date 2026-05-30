index.storage_context.persist("/path to folder/RAG_Index_Storage")
import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
PERSIST_DIR="/path to storage/RAG_Index_Storage"
if not os.path.exists(PERSIST_DIR):
  #load the documents and create the index
  documents=SimpleDirectoryReader('/path to storage/RAG_data').load_data()
  index=VectorStoreIndex.from_documents(documents)
  #store it for later
  index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
  print("I am reading from Meta Storage")
  #load the existing index
  storage_context=StorageContext.from_defaults(persist_dir=PERSIST_DIR)
  index=load_index_from_storage(storage_context)
#either way we can query the index now
#query_engine=index.as_query_engine()
retriever= VectorIndexRetriever(index=index,similarity_top_k=5)
postprocessor=SimilarityPostprocessor(similarity_cutoff=0.77)
query_engine=RetrieverQueryEngine(retriever=retriever,node_postprocessors=[postprocessor])
response=query_engine.query("What is the cause of elevated creatine phosphokinase")
pprint_response(response,show_source=True)
print(response)
