import json
import requests
import random
from dotenv import load_dotenv
from lib.messages import UserMessage, SystemMessage, ToolMessage
from lib.tooling import tool
from lib.llm import LLM

load_dotenv()

def get_random_pokemon():
    BASE_URL = "https://pokeapi.co/api/v2/pokemon?limit=151"
    response = requests.get(BASE_URL)
    response.raise_for_status()
    return random.choice(response.json()['results'])

def get_pokemon_facts(name: str):
    BASE_URL = "https://pokeapi.co/api/v2/pokemon-species/"
    URL = BASE_URL + name
    response = requests.get(URL)
    response.raise_for_status()
    return response.json()['flavor_text_entries'][0]['flavor_text']

@tool
def get_random_pokemon_facts():
    """Gets a random pokemon fact.
        
    Returns:
        dict: Contains name and description of a random pokemon
    """

    random_pokemon = get_random_pokemon()
    name = random_pokemon['name']
    description = get_pokemon_facts(name)
    
    return {"description": name +": "+description}

chat_model_with_tools = LLM(tools=[get_random_pokemon_facts])

messages = [
    SystemMessage(
        content="You are a helpful assistant that can access a tool to get random pokemon facts."
    ),
    UserMessage(content="Can you get me a random pokemon facts?")
]

ai_message = chat_model_with_tools.invoke(messages) # AI realizes it needs function call to complete response and generates how to call function.  

messages.append(ai_message) 

tool_call_id = messages[-1].tool_calls[0].id #AI tells which function to call
tool_name = messages[-1].tool_calls[0].function.name
args = json.loads(messages[-1].tool_calls[0].function.arguments) #AI tells what arguments to pass as input to function


TOOLS = {
    "get_random_pokemon_facts": get_random_pokemon_facts
}

func = TOOLS.get(tool_name)
if not func:
    raise ValueError(f"Unknown tool: {tool_name}")

tool_result = func(**args)

tool_message = ToolMessage(
    content=tool_result["description"], 
    tool_call_id=tool_call_id, 
    name=tool_name
)

messages.append(tool_message)
ai_message = chat_model_with_tools.invoke(messages)
messages.append(ai_message)
response = messages[-1].content
print(response)