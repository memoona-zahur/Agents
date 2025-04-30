import os
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from agents.tool import function_tool
from tavily import TavilyClient
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
today_date = datetime.today()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

config = RunConfig(
    model=model,
    model_provider = client,
    tracing_disabled=True
)

@function_tool("tavily_search")
def tavily_search(query: str):
    """Fetch the latest real-time information regarding strikes, road closures,and routes disruptions that might affect the user's requested route. Provide strike information, including:

    1. City Name

    2. Location (specific road or area affected)

    3.Reason for the strike

    4. Strike position (e.g., road closed, partial disruption, etc.)

    Also, fetch real-time traffic data to identify any closed or affected roads. Provide this information for each query to help the agent deliver accurate, up-to-date route details.
    Additionally, fetch the real-time petrol price per liter. Use this information to help the agent calculate the cost of the journey based on the distance and average fuel consumption. Provide the petrol price in both local currency (PKR) and USD.
    And """
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    response = tavily_client.search(query)
    contents = [item['content'] for item in response.get('results', [])]
    return contents

async def create_agent():
    agent = Agent(
        name="Live_Search_Agent",
        instructions=f"""Your task is to handle user queries by fetching the latest traffic data, including strike-related disruptions, from tools like Tavily_search. When a user asks for a route (e.g., 'find the route from source to destination'), you must:

        1.Provide the best step-by-step route between the source and destination, including road names.

        2. Identify if any roads are blocked due to a strike or disruption and suggest alternative routes.

        3. Provide detailed information on the strike's impact, such as city name, location, reason for the strike, and the position of affected areas, and if no strike information is found then say there is not latest strike update on this route.

        4. If real-time data is unavailable, use historical traffic patterns or suggest typical routes based on usual conditions.

        5. Fetch the current real-time petrol price per liter, and calculate the cost of the journey in both dollars and rupees, based on the route distance and the fuel consumption rate, and at-least give current patrol price if your are not able to calculate the actual total price.

        6. Provide the petrol price in both dollars and rupees per liter.

        Ensure that each step in the route is clear and include all relevant information for each segment of the journey. If there are multiple alternatives, include them with the pros and cons. and also remember today's date {today_date}.And when user not enter date then use today's date as default date.""",
        tools=[tavily_search], # add tools here
        model=model
    )
    return agent

agent = asyncio.run(create_agent())

async def main():
    user_query = input("Enter your query for real-time information (e.g., traffic from Lahore to Karachi): ")
    result = await Runner.run(agent, user_query)
    print("\n--- Live Search Agent Response ---")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())