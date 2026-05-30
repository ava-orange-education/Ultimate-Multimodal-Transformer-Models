!pip install --upgrade llama-index pydantic
!pip install llama-index-llms-huggingface
!pip install -q transformers einops accelerate langchain bitsandbytes
!pip install sentence_transformers
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
documents=SimpleDirectoryReader('/path to folder/RAG_data').load_data()
system_prompt = """
You are a Q & A assistant. You need to answer to the questions as accurately as possible
based on the instructions and context provided
"""
#Default LLAMA-2 Prompt format
query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<ASSISTANT>")
!huggingface-cli login
llm=HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=1000,
    generate_kwargs={"temperature":0.1,"do_sample":False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map='auto',
    model_kwargs={"torch_dtype": torch.float16,"load_in_4bit":True},
)
!pip install -U langchain-community from langchain.embeddings.huggingface import HuggingFaceEmbeddings
!pip install llama-index-embeddings-langchain
from llama_index.core.node_parser import SentenceSplitter
Emb_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
Settings.llm = llm
Settings.embed_model = Emb_model
Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
index=VectorStoreIndex.from_documents(documents,embed_model=Emb_model)
query_engine=index.as_query_engine()
response=query_engine.query("What is SLE assesment,can you explain with a use case and draft an email to my colleague in different hospital")
from llama_index.core.response.pprint_utils import pprint_response pprint_response(response,show_source=True)
print(response)
