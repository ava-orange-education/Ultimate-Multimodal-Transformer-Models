!pip install openai 
import os,openai 
api_key = input("Enter your OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = api_key openai.api_key = api_key
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
)

Emb_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
Settings.llm = llm
Settings.embed_model = Emb_model
Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20) 
Settings.num_output = 2000 
Settings.context_window = 4096
index=VectorStoreIndex.from_documents(documents,embed_model=Emb_model)
query_engine=index.as_query_engine()
response=query_engine.query("What is SLE assesment?, Draft an email in Hindi Language")
pprint_response(response,show_source=True)
print(response)
