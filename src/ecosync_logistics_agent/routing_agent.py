import os
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from agents.run import RunConfig
from dotenv import load_dotenv, find_dotenv
from datetime import datetime

set_tracing_disabled(True)

# Import the Agent instances
from ecosync_logistics_agent.custom_tools.route_planner_agent import agent as route_planner_agent_instance
from ecosync_logistics_agent.custom_tools.live_search_agent import agent as live_search_agent_instance

load_dotenv(find_dotenv())

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

today_date = datetime.today()
current_location = "Lahore, Punjab, Pakistan"

async def create_routing_agent():
    routing_orchestrator_agent = Agent(
        name="Intelligent_Routing_Agent",
        instructions=f"""Your primary goal is to provide the best route from a user-specified source to a destination, considering real-time traffic, potential strikes, and generate a direct link to a Google Maps route showing step-by-step driving directions.

        You have access to the following tools:

        1. get_route: Generates a Google Maps link for the given route.
        2. get_real_time_info: Fetches real-time traffic, strikes, and petrol price information for a given location or route.

        Use these tools to fulfill the user's request. First, get the route link, then fetch real-time information relevant to that route to provide the best possible guidance. Finally, output the route link along with any important alerts and petrol price.
        """,
        tools=[
            route_planner_agent_instance.as_tool(
                tool_name="get_route",
                tool_description="Generates a Google Maps link for the given route (e.g., Lahore -> Karachi).",
            ),
            live_search_agent_instance.as_tool(
                tool_name="get_real_time_info",
                tool_description="Fetches real-time traffic, strikes, and petrol price information for a given location or route (e.g., traffic from Lahore to Karachi, strikes in Karachi today, petrol price in Pakistan).",
            ),
        ],
        model=model
    )
    return routing_orchestrator_agent

# Create the agent instance and export it
agent = asyncio.run(create_routing_agent())

async def main():
    user_query = input("Enter your route request (e.g., I want to go from Lahore to Karachi): ")
    result = await Runner.run(agent, user_query)
    print(f"\n\nFinal response:\n{result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())