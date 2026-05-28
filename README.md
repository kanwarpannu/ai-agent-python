# Readme
The code needs Python 3.11  
To run the code do following steps:  

1. Create env on mac:  
`python3.11 -m venv .venv`  

2. Activate env on mac:  
`source .venv/bin/activate`

3. Install dependencies using following (this includes all requirements across project):  
`pip install -r requirements.txt`

4. Create `.env` file in root project with following details:  
```  
 OPENAI_API_KEY='<api-key-value>'
 OPENAI_BASE_URL='<open-ai-url> like https://api.openai.com/v1 or https://openai.vocareum.com/v1'
```  

## Available projects: 

1. Simple Agent where you can prompt with a specific role:  
`python simple_agent_with_role.py`  

2. A sample agent using internal tool  
`python agent_with_tool.py`  

3. A sample agent using external tool  
`python agent_with_tool_external_api.py`  

4. RAG sample application (Needs chroma db from docker file):  
```
1. docker-compose up -d
2. python agent_with_rag.py
```