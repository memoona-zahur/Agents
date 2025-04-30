import os
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from agents.tool import function_tool
from dotenv import load_dotenv, find_dotenv
from urllib.parse import quote

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

# Utility function to generate a clickable Google Maps route link
def generate_map_link(source, destination, waypoints=None):
    base_url = "https://www.google.com/maps/dir/?api=1"
    origin = quote(source)
    dest = quote(destination)
    url = f"{base_url}&origin={origin}&destination={dest}&travelmode=driving"
    if waypoints:
        wp_string = "|".join([quote(wp) for wp in waypoints])
        url += f"&waypoints={wp_string}"
    return url

@function_tool("route_planner")
def route_planner(user_route: str):
    """Process user input for the route and generate a Google Maps link based on the step-by-step instructions."""

    route_parts = user_route.split('->')
    source = route_parts[0].strip()
    destination = route_parts[-1].strip()
    waypoints = [wp.strip() for wp in route_parts[1:-1]]
    map_link = generate_map_link(source, destination, waypoints)
    return map_link

async def create_agent():
    agent = Agent(
        name="Route_Provider_Agent",
        instructions="""Your task is to take the user’s route input step by step (e.g., 'Lahore -> Multan -> Sukkur -> Karachi').
        You will extract the source, destination, and any waypoints between them, then generate a Google Maps link based on these inputs.
        Ensure that if both cities are in the same country, the agent does not mistakenly consider another country in the route.
        After processing, return a link to the Google Maps route where the user can view the journey step-by-step.""",
        tools=[route_planner],
        model=model
    )
    return agent

agent = asyncio.run(create_agent())

async def main():
    user_query = input("Enter your route (e.g., 'Lahore -> Multan -> Sukkur -> Karachi'): ")
    result = await Runner.run(agent, user_query)
    print("\n--- Route Planner Agent Response ---")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())