import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
retriever= VectorIndexRetriever(index=index,similarity_top_k=3) query_engine=RetrieverQueryEngine(retriever=retriever,node_postprocessors=[postprocessor])
response=query_engine.query("What is the cause of elevated creatine phosphokinase")
pprint_response(response,show_source=True)
print(response)
