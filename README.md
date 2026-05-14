# Readme
The code needs Python 3.11  
To run the code do following steps:  

1. Create env on mac:  
`python3.11 -m venv .venv`  

2. Activate env on mac:
`source .venv/bin/activate`

3. Install dependencies using following:
`pip install -r requirements.txt`

4. Create `.env` file in root project with following details:  
```  
 OPENAI_API_KEY='<api-key-value>'
 OPENAI_BASE_URL='<open-ai-url> like https://api.openai.com/v1 or https://openai.vocareum.com/v1'
```  
4. Now you can run the code using:  
`python simple_agent_with_role.py`  
or  
`python agent_with_tool.py`  