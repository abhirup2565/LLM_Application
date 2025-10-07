# Import relevant functionality
import os,getpass
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
load_dotenv()

#Tools
search = TavilySearch(max_results=2)
search_results = search.invoke("What is the weather in SF")
print(search_results)

# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]

#model
if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
model_with_tools = model.bind_tools(tools)
#we are passing in the model, not model_with_tools. That is because create_react_agent will call .bind_tools for us under the hood.
from langgraph.prebuilt import create_react_agent
agent_executor = create_react_agent(model, tools)

# Create the agent
config = {"configurable": {"thread_id": "abc124"}}
memory = MemorySaver()
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent
input_message = {"role": "user", "content": "Hi!"}
response = agent_executor.invoke({"messages": [input_message]},config)
for message in response["messages"]:
    message.pretty_print()

#  query
input_message = {
    "role": "user",
    "content": "Hi, I'm Bob and I live in San Francisco.",
}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()

# query
input_message = {
    "role": "user",
    "content": "What's the weather where I live?",
}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()

