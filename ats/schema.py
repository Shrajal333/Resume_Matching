import json

def jd_schema():
    with open('job_desc_schema.json', 'r') as file:
        schema = json.load(file)

    return [{
        "type": "function",
        "function": {
            "name": "generate_jd_variants",
            "description": "Generate multiple paraphrased job descriptions similar in nature to the given JD",
            "parameters": schema
        }
    }]