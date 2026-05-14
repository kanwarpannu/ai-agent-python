from dotenv import load_dotenv
from lib.messages import UserMessage, SystemMessage
from lib.llm import LLM

load_dotenv()

chat_model = LLM()

role = input("Please enter LLM role/System message:")
prompt = input("Please enter your prompt:")

messages = [
    SystemMessage(content=role),
    UserMessage(content=prompt)
]
print(messages)
print(type(messages))
response = chat_model.invoke(messages)
print("\nStructured Conversation Response:\n", response.content)