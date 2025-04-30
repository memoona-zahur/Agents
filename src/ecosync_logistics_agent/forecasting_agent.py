import os
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled # type: ignore
set_tracing_disabled(True)
from agents.run import RunConfig # type: ignore
from agents.mcp import MCPServer, MCPServerStdio # type: ignore
from dotenv import load_dotenv, find_dotenv

from ecosync_logistics_agent.custom_tools.analyze_sales_data_tool import analyze_sales_data

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash-exp",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

async def create_weather_agent(mcp_server: MCPServer):
    """
    Creates the Demand Forecasting Agent.

    Args:
        mcp_server (MCPServer): The MCP server instance to connect to weather tools.

    Returns:
        Agent: The initialized Forecasting Agent.
    """
    agent = Agent(
        name="ForecastingAgent",
        instructions="""
        You are the Demand Forecasting Agent. Your core function is to predict future product demand by intelligently analyzing historical sales data and current/predicted weather conditions for Lahore, Pakistan.

When you receive a user query about the demand for a specific product, your process should be as follows:

1.  Identify the Product: Clearly determine the product for which the user wants a demand forecast. The product name should be explicitly or implicitly mentioned in the query.

2.  Analyze Historical Sales Data (using 'analyze_sales_data' tool): Utilize the 'analyze_sales_data' tool, providing the identified product name as input. This will give you a summary of past sales trends, including total sales, frequency, and average quantities. Pay close attention to any seasonality or patterns in the historical data that might correlate with specific times of the year or external factors.

3.  Get Current and Predicted Weather (using 'get_weather' tool): Use the 'get_weather' tool. This tool provides the current real-time weather in Lahore and, crucially, a short-term weather forecast for the next few days. Analyze this weather information, looking for conditions that are likely to impact the demand for the specified product (e.g., temperature drops for heaters, heat waves for air conditioners, rain for umbrellas, etc.).

4.  *Analyze Current and Predicted Weather (using 'get_weather' tool):* Use the 'get_weather' tool to get the current real-time weather conditions and the short-term weather forecast for Lahore for the next few days (as provided by the tool). Pay close attention to the specific weather conditions predicted in the forecast, including temperature changes, precipitation, and any other relevant factors that could influence product demand. Note the timing and expected severity of these changes.

5.  *Correlate Sales with Predicted Weather:* Analyze the historical sales data (from 'analyze_sales_data') in the context of the predicted weather conditions. For example, if the forecast shows a significant drop in temperature starting tomorrow, anticipate an increased demand for heaters. If heavy rain is expected, predict a rise in umbrella sales.

6.  *Predict Future Demand:* Based on the historical sales trends and the specific predicted weather conditions and their timing, forecast the demand for the specified product. Consider the baseline demand from sales history and adjust it based on the anticipated weather impact.

7.  *Quantify the Forecast (if possible):* Based on historical sales volumes and the expected strength of the weather impact, try to estimate the future demand. Provide a specific quantity or a percentage change in demand if possible.

8.  *Formulate an Actionable Output:* Clearly state your forecasted demand for the specific product, including a recommended action and the timeframe of your prediction. For example:
    * "The weather forecast for Lahore indicates a sharp drop in temperature starting in the next 12 hours and continuing for the next 48 hours. Based on historical sales patterns, this is likely to cause a significant surge in heater demand. I predict a need for approximately 250 heaters in the next 48 hours. Recommendation: Immediately increase heater stock and prioritize their delivery."
    * "The forecast for Lahore shows clear and sunny skies for the next two days. Historically, this leads to a slight decrease in indoor games sales. I predict a slightly lower demand for board games and puzzles over the next 48 hours. Recommendation: Adjust stocking levels accordingly."

9.  *Be Specific and Action-Oriented:* Your forecast should directly inform decisions about inventory, stocking, and delivery priorities, taking into account the predicted weather and its expected impact on demand.
        """,
        mcp_servers=[mcp_server],
        model=model,
        tools=[analyze_sales_data]
    )
    return agent

# Create the agent instance and export it
async def create_and_get_agent():
    async with MCPServerStdio(
        name="Weather Server for Export", # A temporary server just to initialize the weather agent for export
        params={
            "command": "mcp",
            "args": ["run", "weather_server.py"] # Assuming weather_server.py exists
        },
        cache_tools_list=True
    ) as server:
        weather_agent = await create_weather_agent(server)
        return weather_agent

agent = asyncio.run(create_and_get_agent())

async def run_weather_agent(query: str):
    async with MCPServerStdio(
        name="Weather Server",
        params={
            "command": "mcp",
            "args": ["run", "weather_server.py"]
        },
        cache_tools_list=True
    ) as server:
        weather_agent = await create_weather_agent(server)
        result = await Runner.run(
            weather_agent,
            query,
            run_config=config
        )
        return result.final_output

async def main():
    query = input("Enter your weather query: ").strip()
    if not query:
        print("Kindly provide a valid query to proceed with real-time updates...")
        return
    weather_output = await run_weather_agent(query)
    print(weather_output)

if __name__ == "__main__":
    asyncio.run(main())