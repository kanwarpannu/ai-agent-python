import json
from dotenv import load_dotenv
from lib.messages import UserMessage, SystemMessage, ToolMessage
from lib.tooling import tool
from lib.llm import LLM

load_dotenv()

@tool
def get_weather(city: str):
    """Get the current temperature for a city.
    
    Args:
        city (str): Name of the city to check weather for
        
    Returns:
        dict: Contains temperature information for the requested city
    """
    # In a real application, this would call a weather API
    mock_weather = {
        "São Paulo": "28°C",
        "Oslo": "-3°C",
        "New York": "15°C",
        "Tokyo": "22°C"
    }
    return {"temperature": mock_weather.get(city, "Unknown")}

chat_model_with_tools = LLM(tools=[get_weather])

messages = [
    SystemMessage(
        content="You are a helpful assistant that can access a tool to get current temperature " 
                "for cities. Use the tool whenever someone asks about the weather or temperature " 
                "in a specific location. Infor the user if you don't know the answer."
    ),
    UserMessage(content="How cold is it in New York?")
]

ai_message = chat_model_with_tools.invoke(messages) # AI realizes it needs function call to complete response and generates how to call function.  

messages.append(ai_message) 

tool_call_id = messages[-1].tool_calls[0].id #AI tells which function to call
tool_name = messages[-1].tool_calls[0].function.name
args = json.loads(messages[-1].tool_calls[0].function.arguments) #AI tells what arguments to pass as input to function


TOOLS = {
    "get_weather": get_weather
}

func = TOOLS.get(tool_name)
if not func:
    raise ValueError(f"Unknown tool: {tool_name}")

tool_result = func(**args)

tool_message = ToolMessage(
    content=tool_result["temperature"], 
    tool_call_id=tool_call_id, 
    name=tool_name
)

messages.append(tool_message)
ai_message = chat_model_with_tools.invoke(messages)
messages.append(ai_message)
response = messages[-1].content
print(response)