from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langsmith import traceable
from langchain_tavily import TavilySearch
# from tavily import TavilyClient

load_dotenv()

# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)


llm = ChatOllama(model="qwen3:0.6b", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


@traceable
def main():
    print("Hello from langchain-course!")

    result = agent.invoke(
        {"messages": HumanMessage(content="Search for 3 job postings for an ai engineer using langchain. in the bay area on linkedin and list their details.")}
    )
    print(result)


if __name__ == "__main__":
    main()
