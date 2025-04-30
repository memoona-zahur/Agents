from mcp.server.fastmcp import FastMCP # type: ignore
import requests

# create server
mcp = FastMCP("Weather Server")

@mcp.tool()
def get_weather(city: str)->str:
    """
    First of all, give current weather information.
    Fetches the current real-time weather conditions and a forecast for the next 24 hours for the specified city.
    Returns a text-based report including temperature, conditions, wind, humidity, and a brief outlook for the next day.
    """
    endpoint = "https://wttr.in"
    response = requests.get(f"{endpoint}/{city}")
    return response.text

# run the server
if __name__ == "__main__":
    mcp.run()