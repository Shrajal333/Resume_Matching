import json

def schema_tool():
    with open('resume_schema.json', 'r') as file:
        schema = json.load(file)

    return [{
        "type": "function",
        "function": {
            "name": "extract_resume_info",
            "description": "Extract structured information from a candidate's resume",
            "parameters": schema
        }
    }]