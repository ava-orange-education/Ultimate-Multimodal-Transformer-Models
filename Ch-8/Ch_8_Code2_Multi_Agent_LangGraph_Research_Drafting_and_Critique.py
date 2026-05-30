!pip install python-dotenv
from dotenv import load_dotenv, find_dotenv
_=load_dotenv(find_dotenv())
!apt update
!apt install graphviz-dev -y
!pip install langgraph langchain langchain_openai langchain_community tavily-python langgraph-checkpoint-sqlite pygraphviz
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
#Prepares an in-memory checkpoint to store and recover current states during the multi-step workflow.
memory = SqliteSaver.from_conn_string(":memory:")
class AgentState(TypedDict):
    task: str   #Human Input
    plan: str   #Planning Agent
    draft: str   #Draft of essay
    critique: str   #Critique Agent
    content: List[str]   #Tavily research
    revision_number: int
    max_revisions: int  #criteria to stop
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o", temperature=0)
#Tells the LLM to create an outline for the essay based on user input.
PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay.Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes or instructions for the sections."""
#Instructs the LLM how to draft or revise the essay using available outline and research.
WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.Generate the best essay possible for the user's request and the initial outline.If the user provides critique, respond with a revised version of your previous attempts.Utilize all the information below as needed:
------
{content}"""
#Tells the LLM to critique and suggest improvements.
REFLECTION_PROMPT = """You are a teacher grading an essay submission.Generate critique and recommendations for the user's submission.Provide detailed recommendations, including requests for length, depth, style, etc."""
#Asks LLM to create precise search queries for relevant research at each iteration.
RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can be used when writing the following essay. Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""
RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can be used when making any requested revisions (as outlined below). Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""
from pydantic.v1 import BaseModel
class Queries(BaseModel):
    queries: List[str]
from tavily import TavilyClient
import os
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
#This node is useful for transforming high-level tasks into structured actionable instructions within an agentic AI workflow.
def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {"plan": response.content}
def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state.get('content') or [] #It initializes content as the existing 
    content in the state (state['content']) or an empty list if none exists.
    #This allows accumulating results over multiple runs or queries.
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2) 
        #For each query, it calls the Tavily search tool
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}
#Write or Revise the Essay, uses collected research and the outline to write the essay (or revise if already written).Increments the revision counter.
def generation_node(state: AgentState):
    content = "\n\n".join(state.get('content') or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)),user_message]
    response = model.invoke(messages) #Calls the language model with the 
    constructed messages to generate a response.
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1
    }
#This node adds a reflection and self-improvement step.
def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}

#Guide Iterative Research,based on critique feedback, LLM creates new targeted research queries.Aggregates new findings with previous ones for the next essay revision.
def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])])
    content = state.get('content') or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}
#Once the allowed number of iterations/revisions is reached, ends the loop.Otherwise, returns to reflection (review and revise again).
def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"
builder = StateGraph(AgentState)
builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)
builder.set_entry_point("planner")
builder.set_entry_point("planner")
builder.add_conditional_edges(
    "generate",
    should_continue,
    {END: END, "reflect": "reflect"}
)
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
from IPython.display import Image
Image(graph.get_graph().draw_png())
thread = {"configurable": {"thread_id": "1"}}
for s in graph.stream({
    'task': "what is the difference between LangGraph and MCP, will MCP 
     take over LangGraph framework in long run?","max_revisions": 2,
     "revision_number": 1,}, thread):
    print(s)
{'planner': {'plan': "**Title: Comparative Analysis of LangGraph and MCP: Will MCP Supersede LangGraph in the Long Run?**\n\n**I. Introduction**\n   - Brief introduction to the importance of frameworks in computational linguistics and machine learning.\n   - Present the purpose of the essay: to compare LangGraph and MCP, and evaluate the potential for MCP to replace LangGraph in the future.\n  
.. .. .. ..
Will MCP Take Over LangGraph?**\n\n**A. Arguments for MCP Superseding LangGraph**\n\nMCP's scalability, flexibility, and advanced communication protocols position it as a strong contender to replace LangGraph, particularly in industries where interoperability is paramount. Its ability to integrate diverse tools and systems offers superior solutions in scenarios requiring dynamic and adaptable frameworks.\n\n**B. Counterarguments**\n\nDespite MCP's advantages, LangGraph's established strengths, such as its ease of use and robust orchestration capabilities, may allow it to maintain its position in the market. Its specific strengths and potential for adaptation could ensure its continued relevance in certain applications.\n\n**VII. Conclusion**\n\nIn conclusion, both LangGraph and MCP offer valuable contributions to the field of computational linguistics and machine learning. While MCP's scalability and flexibility present a compelling case for its potential to supersede LangGraph, the latter's established user base and proven effectiveness in specific applications suggest that it will continue to hold its ground. Ultimately, the future of these frameworks will depend on technological advancements, market demands, and industry needs, shaping the landscape of language processing and machine communication.", 'revision_number': 3}}
