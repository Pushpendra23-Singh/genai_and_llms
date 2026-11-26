"""
LangChain Agent with Calculator, Search & Memory
pip install langchain langchain-community ollama duckduckgo-search numexpr
Requires: ollama pull mistral
"""
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.llms import Ollama
from langchain.tools import DuckDuckGoSearchRun
import numexpr as ne

tools = [
    Tool(name="Calculator", func=lambda x: str(ne.evaluate(x)),
         description="Evaluates a math expression like '2+2' or '10*5'."),
    Tool(name="Search", func=DuckDuckGoSearchRun().run,
         description="Search the web for current information."),
]

agent = initialize_agent(
    tools=tools,
    llm=Ollama(model="mistral"),
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    verbose=True,
)

queries = [
    "What is 25 * 48?",
    "Who invented the telephone?",
    "What is 100 / 4 + 10?",
]
for q in queries:
    print(f"\nUser: {q}\nAgent: {agent.run(q)}")
