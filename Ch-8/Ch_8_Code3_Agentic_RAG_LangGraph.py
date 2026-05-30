!pip install -q langchain==0.2.9 
!pip install -q langchain-openai==0.1.7 
!pip install -q langchain-chroma==0.1.2 
!pip install -q langgraph==0.0.55 
!pip install -q beautifulsoup4==4.12.2 
!pip install -q tavily-python==0.3.5 
!pip install -q python_dotenv==1.0.1 
!pip install -q langchain-experimental==0.0.62 
!pip install -q simple-colors==0.1.5 
!pip install -q pypdf==3.16.2 
!pip install -q rank_bm25==0.2.2
!git clone https://github.com/genaiconference/RAG_Workshop_DHS2024.git 
import os
import openai
_key = input("Enter your OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = api_key
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
llm = ChatOpenAI(model_name="gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
from google.colab import drive
drive.mount('/content/drive')
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader('/path_to_folder/Leave-Policy-India.pdf')
documents = loader.load()
def chunkByWord(text):
    return len(text.split(" "))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, length_function = chunkByWord, chunk_overlap=50, add_start_index = False)
#The length_function overrides the default character-based length measurement.
splits = text_splitter.split_documents(documents)
from langchain.vectorstores import Chroma
persist_directory = os.getcwd() +'/vectorstore/leave_policy/'
# Create the vector store
vectordb = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory )
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
from langchain.document_loaders import DirectoryLoader
loader = PyPDFLoader('/path_to_folder/Microsoft_2023.pdf')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, length_function = chunkByWord, chunk_overlap=50, add_start_index = False)
docs = text_splitter.split_documents(documents)
#save to disk
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory='vectorstore/annual_reports/')
from langchain.retrievers import MergerRetriever, BM25Retriever
# Define 2 different retrievers with 2 different search type for diversity
retriever_1 = vectorstore.as_retriever(search_type="similarity",
                                                 search_kwargs={"k": 5})
retriever_2 = vectorstore.as_retriever(search_type="mmr",
                                                 search_kwargs={"k": 5}) 
#Applies Maximal Marginal Relevance to ensure diversity among retrieved chunks.
# BM25 Retriver for keyword based retrieval using BM25 algorithm
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3 # set the num of docs to be retrieved
# Hybrid Retriver using Semantic Search and Keyword search together
lotr_annual_reports = MergerRetriever(retrievers=[retriever_1, retriever_2, bm25_retriever])
from langchain.tools.retriever import create_retriever_tool
leave_policy_tool = create_retriever_tool(retriever=retriever,
name = 'Leave_Policy_Data_Tool', description="Use this tool to answer questions related to the leave policies of the company")
annual_reports_tool = create_retriever_tool(retriever=lotr_annual_reports,
name = 'Annual_Reports_Data_Tool', description="Use this tool to answer questions related to the annual reports of Microsoft")
api_key = input("Enter your TAVILY API key: ")
os.environ["TAVILY_API_KEY"] = api_key
import json
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_community.utilities import BingSearchAPIWrapper
from langchain.tools.retriever import create_retriever_tool
from langchain_community.retrievers import TavilySearchAPIRetriever
from RAG_Workshop_DHS2024.prompts import *
from RAG_Workshop_DHS2024.copilot_utils import create_qa_agent, create_chat_agent
# Create a tool to processes content from a webpage,a tool that fetches an HTML page using the requests library and parses it with BeautifulSoup.
#It extracts the page text and recursively splits it into manageable chunks.
@tool("process_content", return_direct=False)
def process_content(url: str, chunk_size: int = 10000) -> str:
    """
    Use this tool to extract content from HTML pages and chunk it recursively.
    Args:
      url (str): The URL of the HTML page to process.
      chunk_size (int, optional): The maximum size of each text chunk. Defaults to 
      100000.
    Returns:
      str: The chunked text content extracted from the HTML page.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser',from_encoding="iso-8859-
    1")
    text = soup.get_text()
    def chunk_text(text, chunk_size):
        if len(text) <= chunk_size:
            return text
        else:
            midpoint = len(text) // 2
            left_chunk = chunk_text(text[:midpoint], chunk_size)
            right_chunk = chunk_text(text[midpoint:], chunk_size)
            return left_chunk + "\n" + right_chunk
        chunks = chunk_text(text, chunk_size)
        return chunks
#A tool is defined to query Bing (through BingSearchAPIWrapper) when current or real-time information is needed.
@tool("bing_search", return_direct=False)
def bing_search(query: str) -> str:
    """Use this tool when you need to answer questions related to current events  
    and latest happenings"""
    bing_search = BingSearchAPIWrapper()
    results = bing_search.results(query, 5)
    return results if results else "No results found."
# Create a tool to search the internet using TavilySearchAPI - Alternative to Bing Search
def get_tavily_search_retriever():
    retriever = TavilySearchAPIRetriever(k=7)
    retriever_description = "Use this tool when you need to answer questions related to current events and latest happenings or anything related to specific year"
    retrieve_tool = create_retriever_tool(retriever=retriever, name="internet_search", description=retriever_description)
    return retrieve_tool
from simple_colors import *
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from IPython.display import Markdown
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.output_parsers import JsonOutputParser
class FinanceCheck_class(BaseModel):
    """Binary score for finance check on the user question."""
    binary_score: str = Field(description="Given a user question, return 'Yes' if is related to Finance/Accounting/Reporting else 'No'")
#Each function represents a “node” in a state graph that processes a user question:
def get_finance_check(_llm, META_RESPONSE_PROMPT, state):
    parser = PydanticOutputParser(pydantic_object=FinanceCheck_class)
    prompt = PromptTemplate(template=META_RESPONSE_PROMPT,
                            input_variables=["question"],
                            partial_variables={"format_instructions": 
    parser.get_format_instructions()},)
    prompt_and_model = prompt | llm
    output = prompt_and_model.invoke({"question": question})
    result = parser.invoke(output)
    return result.binary_score
def finance_check_node(state):
    """Use this tool to check whether the user question is about   
    finance or accounting."""
    observation = get_finance_check(llm, META_RESPONSE_PROMPT_DETAILED, 
    state['question'])
    print("------ENTERING: FINANCE CHECK NODE------")
    print(f"------OBSERVATION: {observation}------")
    return {"finance_check": observation}
def generic_response_node(state):
    """Use this tool to answer user generic questions"""
    print("------ENTERING: GENERIC RESPONSE NODE------")
    answer = llm.invoke(META_ANSWER_PROMPT.format(question=question))
    final_answer = answer.content
    display(Markdown(f"""<font color="black">**GENERIC RESPONSE:**</font>"""))
    display(Markdown(f"""<font color="green">{final_answer}</font>"""))
    return {"generic_response": answer, "final_answer": final_answer}
#Routes the question based on its nature—either to internal data (e.g., finance) or to web search.
class RouteQuery_class(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["FINANCE", "WEB"] = Field(description="Given a user   
    question choose to route it to WEB or FRA.")
def get_query_rerouter_check(llm, QUERY_REROUTER_PROMPT, state):
    parser = PydanticOutputParser(pydantic_object=RouteQuery_class)
    prompt = PromptTemplate(template=QUERY_REROUTER_PROMPT,
                            input_variables=["question"],
                            partial_variables={"format_instructions":   
    parser.get_format_instructions()},)
    prompt_and_model = prompt | llm
    output = prompt_and_model.invoke({"question": question})
    query_rerouter_check = parser.invoke(output)
    return query_rerouter_check.datasource
def query_rerouter_node(state):
    """Use this tool to check whether the user question needs an internet search 
    to answer"""
    print("------ENTERING: QUERY REROUTER NODE------")
    observation = get_query_rerouter_check(llm, QUERY_REROUTER_PROMPT_DETAILED, 
    state['question'])
    print(f"------OBSERVATION: {observation}------")
    return {"query_rerouter_check": observation}
#prompts the user to decide if an internet search should be performed if internal data is insufficient.
def human_check_node(state):
    user_input = input("The answer is not available from the private data sources!  
    Do you want me to search over the internet to answer?")
    return {"human_response": user_input.lower()}
def get_web_search_answer(state):
    internet_search_tool = get_tavily_search_retriever()
    # tools = [bing_search, process_content]
    tools = [internet_search_tool, process_content]
    generate_prompt = """You are a web searcher trained to retrieve the current events from the internet. Search the internet for information. Generate the best answer possible for the user's request with mandatory mention of the sources and the hyperlinks for the sources wherever it is possible. Think step by step. Breakdown the question if it has multiple tasks and finally merge your results.
Always crave for the best version of answer. Before giving the final answer, try another method. Then reflect on the answers of the two methods you did and ask yourself if it answers correctly the original question. If you are not sure, try another method. If the methods tried do not give the same result, reflect and try again until you have two methods that have the same result. If you are sure of the correct answer, create a beautiful and thorough response.
** DO NOT MAKE UP AN ANSWER OR USE YOUR PRIOR KNOWLEDGE, ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE **
PLEASE NOTE THAT IF NO SPECIFIC YEAR MENTIONED IN THE QUESTION, ALWAYS LOOK FOR THE LATEST YEAR. """
    generate_agent = create_qa_agent(llm, tools, generate_prompt, verbose=False)
    answer = generate_agent.invoke({"input": state['question']})
    return answer['output']
def web_search_node(state):
    """Use this tool when you need to answer questions related to current events 
    and latest happenings"""
    print("------ENTERING: WEB SEARCH NODE------")
    ## Write code here to get answer
    response = get_web_search_answer(state)
    print(f"------WEB SEARCH ANSWER: {response}------")
    display(Markdown(f"""<font color="Black">**WEB ANSWER:**</font>"""))
    display(Markdown(f"""<font color="green">{response}</font>"""))
    return {"web_response": response, "final_answer": response}
def private_internal_data_answer_node(state):
    """Use this tool to answer any questions related to leave policies of the compny"""
    print("------ENTERING: PRIVATE INTERNAL DATA ANSWER NODE------")
    tools = [leave_policy_tool, annual_reports_tool]
    generate_agent = create_qa_agent(llm, tools, private_internal_data_prompt, verbose=False)
    answer = generate_agent.invoke({"input": state['question']})
    final_answer = f"""**PRIVATE INTERNAL DATA ANSWER:** {answer['output']} \n\n"""
    display(Markdown(f"""<font color="green">{final_answer}</font>"""))
    return {"private_internal_data_response": answer['output'], 'final_answer':final_answer}
def overall_status_check_node(state):
    """Use this tool to check the overall status and update the config settings"""
    print("------ENTERING: OVERALL STATUS CHECK NODE------")
    final_answer = state['final_answer']
    observation = "Completed"
    print("-------------------------------- {} --------------------------------".format(black("Final Answer", ['bold'])))
    display(Markdown(f"""**FINAL ANSWER:**\n\n <font 
    color="green">{final_answer}</font>"""))
    return {"overall_status_check": observation, 'final_answer':final_answer}
from typing_extensions import TypedDict
from typing import Annotated, List
import operator
class GraphState(TypedDict):
    """Represents the state of our graph.
      Attributes:
      question: User Question
      finance_check: Whether the user question is about finance or generic 
      question
      query_rerouter_check: Check to assess the questions needs internal private 
      data sources or seb search to answer
      source_detector_check: Check to assess what internal private sources to be 
      needed to answer user question
      overall_status_check: Check to assess the overall graph completion
      generic_response: Generic answer
      human_response: Human response to search over internet
      web_response: Web search response
      private_internal_data_response: IFRS Response
      final_answer: Final Answer
    """question: str
    finance_check: str
    query_rerouter_check: str
    source_detector_check: List[str]
    overall_status_check: str
    generic_response: str
    human_response: str
    web_response: str
    private_internal_data_response: str
    final_answer: Annotated[str, operator.add]
from langgraph.graph import StateGraph
#A StateGraph builder is used to add each node (function) and specify the order of execution.
builder = StateGraph(GraphState)
builder.add_node("finance_check_node", finance_check_node)
builder.add_node("generic_response_node", generic_response_node)
builder.add_node("query_rerouter_node", query_rerouter_node)
builder.add_node("human_check_node", human_check_node)
builder.add_node("web_search_node", web_search_node)
builder.add_node("private_internal_data_answer_node", private_internal_data_answer_node)
builder.add_node("overall_status_check_node", overall_status_check_node)
from langgraph.checkpoint.sqlite import SqliteSaver
builder.set_entry_point("finance_check_node")
#Based on node outputs (e.g., whether the question is finance-related), the graph decides which node to process next.
def finance_check_edge(state):
    if state["finance_check"] == "Yes":
        print("------DECISION: INVOKE QUERY REROUTER NODE------")
        return "query_rerouter_node"
    else:
        print("------DECISION: INVOKE GENERIC RESPONSE NODE------")
        return "generic_response_node"
finance_check_possible_nodes = ["query_rerouter_node", "generic_response_node"]
builder.add_conditional_edges("finance_check_node", finance_check_edge, finance_check_possible_nodes)
builder.add_edge("generic_response_node", "overall_status_check_node")
def query_rerouter_edge(state):
    if state["query_rerouter_check"] == "FINANCE":
        print("------DECISION: INVOKE SOURCE DETECTOR NODE------")
        return "private_internal_data_answer_node"
    else:
        print("------DECISION: CHECK WITH USER TO INVOKE WEB SEARCH NODE------")
        return "human_check_node"
query_rerouter_possible_nodes = ["human_check_node", "private_internal_data_answer_node"]
builder.add_conditional_edges("query_rerouter_node", query_rerouter_edge, query_rerouter_possible_nodes)
def human_check_edge(state):
    if state['human_response'] == "yes":
        print("------DECISION: INVOKE WEB SEARCH NODE------")
        return "web_search_node"
    else:
        print("------Your question cannot be answered due to lack of information from private data sources, please ask a different question!------")
        return "overall_status_check_node"
builder.add_conditional_edges("human_check_node", human_check_edge, ["web_search_node", "overall_status_check_node"])
builder.add_edge("web_search_node", "overall_status_check_node")
builder.add_edge("private_internal_data_answer_node", "overall_status_check_node")
builder.set_finish_point("overall_status_check_node")
memory = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(checkpointer=memory)
#The graph starts at the finance_check_node and ends at the overall_status_check_node.
#The state graph is compiled with a checkpointing mechanism (using an in-memory SQLite saver) to track the workflow.
def run_graph(question, thread_id):
    inputs = {"question": question}
    thread = {"configurable": {"thread_id": str(thread_id)}}
    for output in graph.stream(inputs, thread, stream_mode='values'):
        pass
question = "What is the income of Dr Reddys Labs in 2024 Q1?"
run_graph(question,9)
------ENTERING: FINANCE CHECK NODE------
------OBSERVATION: Yes------
------DECISION: INVOKE QUERY REROUTER NODE------
------ENTERING: QUERY REROUTER NODE------
------OBSERVATION: WEB------
------DECISION: CHECK WITH USER TO INVOKE WEB SEARCH NODE------
The answer is not available from the private data sources! Do you want me to search over the internet to answer?Yes
------DECISION: INVOKE WEB SEARCH NODE------
------ENTERING: WEB SEARCH NODE------
------WEB SEARCH ANSWER: In the first quarter of the fiscal year 2024 (Q1 FY24), Dr. Reddy's Laboratories reported a revenue from operations of ₹6,757.9 crore, which represents a 13.88% year-on-year increase. The profit after tax (PAT) for this quarter was ₹1,392.4 crore, showing a slight decrease of 0.90% compared to the same quarter in the previous year.
For more details, please refer to the sources:
- [Business Standard](https://www.business-standard.com/)
- [Livemint](https://www.livemint.com/)------
WEB ANSWER:
In the first quarter of the fiscal year 2024 (Q1 FY24), Dr. Reddy's Laboratories reported a revenue from operations of ₹6,757.9 crore, which represents a 13.88% year-on-year increase. The profit after tax (PAT) for this quarter was ₹1,392.4 crore, showing a slight decrease of 0.90% compared to the same quarter in the previous year.
For more details, please refer to the sources:
Business Standard
Livemint
------ENTERING: OVERALL STATUS CHECK NODE------
-------------------------------- Final Answer --------------------------------FINAL ANSWER:
In the first quarter of the fiscal year 2024 (Q1 FY24), Dr. Reddy's Laboratories reported a revenue from operations of ₹6,757.9 crore, which represents a 13.88% year-on-year increase. The profit after tax (PAT) for this quarter was ₹1,392.4 crore, showing a slight decrease of 0.90% compared to the same quarter in the previous year.
For more details, please refer to the sources:
Business Standard
Livemint
question = "How many times i can avail parental leave policy"
run_graph(question, 92)
------ENTERING: FINANCE CHECK NODE------
------OBSERVATION: Yes------
------DECISION: INVOKE QUERY REROUTER NODE------
------ENTERING: QUERY REROUTER NODE------
------OBSERVATION: FINANCE------
------DECISION: INVOKE SOURCE DETECTOR NODE------
------ENTERING: PRIVATE INTERNAL DATA ANSWER NODE------
PRIVATE INTERNAL DATA ANSWER: - Maternity Leave: Eligible for a total of 26 weeks twice in the service tenure, as per the Maternity Benefit Act and Maternity Benefit (Amendment) Bill 2017. Additionally, women are entitled to 6 extra weeks, totaling 32 weeks for 2 surviving children. For a third surviving child, the leave is 12 weeks. Maternity leave cannot be accumulated.
Paternity Leave: Eligible for 7 working days, which is not accumulable or encashable. It can be availed for up to 2 children only.
------ENTERING: OVERALL STATUS CHECK NODE------
-------------------------------- Final Answer --------------------------------FINAL ANSWER:
PRIVATE INTERNAL DATA ANSWER: - Maternity Leave: Eligible for a total of 26 weeks twice in the service tenure, as per the Maternity Benefit Act and Maternity Benefit (Amendment) Bill 2017. Additionally, women are entitled to 6 extra weeks, totaling 32 weeks for 2 surviving children. For a third surviving child, the leave is 12 weeks. Maternity leave cannot be accumulated.
Paternity Leave: Eligible for 7 working days, which is not accumulable or encashable. It can be availed for up to 2 children only.
question = "List down all the microsoft products and the corresponding profits for the year 2023"
run_graph(question,90)
------ENTERING: FINANCE CHECK NODE------
------OBSERVATION: Yes------
------DECISION: INVOKE QUERY REROUTER NODE------
------ENTERING: QUERY REROUTER NODE------
------OBSERVATION: FINANCE------
------DECISION: INVOKE SOURCE DETECTOR NODE------
------ENTERING: PRIVATE INTERNAL DATA ANSWER NODE------
PRIVATE INTERNAL DATA ANSWER: As per the Annual Report for the year ended June 30, 2023, Microsoft reported the following revenue growth and changes for its major product lines:
Microsoft Cloud Revenue: Increased by 22% to $111.6 billion.
Office Commercial Products and Cloud Services: Revenue increased by 10%, with Office 365 Commercial growth of 13%.
Office Consumer Products and Cloud Services: Revenue increased by 2%, with Microsoft 365 Consumer subscribers rising to 67 million.
LinkedIn: Revenue increased by 10%.
Dynamics Products and Cloud Services: Revenue increased by 16%, driven by Dynamics 365 growth of 24%.
Server Products and Cloud Services: Revenue increased by 19%, driven by Azure and other cloud services growth of 29%.
Windows OEM: Revenue decreased by 25%.
Devices: Revenue decreased by 24%.
Windows Commercial Products and Cloud Services: Revenue increased by 5%.
Xbox Content and Services: Revenue decreased by 3%.
Search and News Advertising: Revenue excluding traffic acquisition costs increased by 11%.
These figures provide a comprehensive overview of the performance and profitability of Microsoft's major product lines for the year 2023. 
Citations: 
Annual Report, Part II, Item 7, Page 36, 39, 41.
------ENTERING: OVERALL STATUS CHECK NODE------
-------------------------------- Final Answer --------------------------------
FINAL ANSWER:
PRIVATE INTERNAL DATA ANSWER: As per the Annual Report for the year ended June 30, 2023, Microsoft reported the following revenue growth and changes for its major product lines:
Microsoft Cloud Revenue: Increased by 22% to $111.6 billion.
Office Commercial Products and Cloud Services: Revenue increased by 10%, with Office 365 Commercial growth of 13%.
Office Consumer Products and Cloud Services: Revenue increased by 2%, with Microsoft 365 Consumer subscribers rising to 67 million.
LinkedIn: Revenue increased by 10%.
Dynamics Products and Cloud Services: Revenue increased by 16%, driven by Dynamics 365 growth of 24%.
Server Products and Cloud Services: Revenue increased by 19%, driven by Azure and other cloud services growth of 29%.
Windows OEM: Revenue decreased by 25%.
Devices: Revenue decreased by 24%.
Windows Commercial Products and Cloud Services: Revenue increased by 5%.
Xbox Content and Services: Revenue decreased by 3%.
Search and News Advertising: Revenue excluding traffic acquisition costs increased by 11%.
These figures provide a comprehensive overview of the performance and profitability of Microsoft's major product lines for the year 2023. 
Citations: 
Annual Report, Part II, Item 7, Page 36, 39, 41.
