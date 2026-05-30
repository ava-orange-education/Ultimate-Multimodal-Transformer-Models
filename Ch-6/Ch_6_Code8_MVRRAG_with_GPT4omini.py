!pip install langchain openai pypdf faiss-gpu tiktoken
!pip install llama-index
from google.colab import drive
drive.mount('/content/drive')
!pip install -U langchain-community
!pip install -U langchain-openai
!pip install chromadb
import os
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores import Chroma 
from langchain.storage import InMemoryStore 
from langchain.retrievers import MultiVectorRetriever
from langchain.docstore import InMemoryDocstore 
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import uuid
from google.colab import drive
drive.mount('/content/drive')
api_key = input("Enter your OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = api_key
loader = PyPDFLoader('/path_to_folder/Leave-Policy-India.pdf')
documents = loader.load()
for item in documents:
    item.metadata['source_type'] = 'Parent'
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="full_documents", embedding_function=embeddings)
# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)
doc_ids = [str(uuid.uuid4()) for _ in documents]
# Doc ID are assigned to each Parent.
retriever.docstore.mset(list(zip(doc_ids, documents)))
#Create Child documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
# The splitter to use to create smaller chunks
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, separators=["\n\n","\n"])
sub_docs = []
for i, doc in enumerate(documents):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
    sub_docs.extend(_sub_docs)
for item in sub_docs:
    item.metadata['source_type'] = 'Children'
retriever.vectorstore.add_documents(sub_docs)
print("Successfully added children to the vector store")
from langchain.schema.document import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
# Prepare the summaries created in document format
summary_docs = [
    Document(page_content=s, metadata={"source_type" : "summary",id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]
# Add the summary documents to the vector store
retriever.vectorstore.add_documents(summary_docs)
print("Successfully added summaries to the vector store")
functions = [
    {
        "name": "hypothetical_questions",
        "description": "Generate hypothetical questions",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["questions"],
        },
    }
]
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template(
        "Generate a list of exactly 3 hypothetical questions that the below document could be used to answer:\n\n{doc}"
    )
    | llm.bind(
        functions=functions, function_call={"name": "hypothetical_questions"}
    )
    | JsonKeyOutputFunctionsParser(key_name="questions")
)
hypothetical_questions = chain.batch(documents, {"max_concurrency": 5})
question_docs = []
for i, question_list in enumerate(hypothetical_questions):
    question_docs.extend(
        [Document(page_content=s, metadata={"source_type":"Hypothetical Question", id_key: doc_ids[i]}) for s in question_list] )
retriever.vectorstore.add_documents(question_docs)
print("Successfully added hypothetical questions to the vector store")
from utils import pretty_print_result
query = "How many times can a male employee avail paternity leave? Can you explain in English"
pretty_print_result(query, llm, retriever)
