import json

def row_to_text(row):
    fields = [
        f"Experience: {row['Experience Details']}",
        f"Projects: {row['Projects']}",
        f"Awards: {row['Awards']}",
        f"Certificates: {row['Certificates']}",
        f"Publications: {row['Publications']}",
        f"Skills - Languages: {row['Skills - Languages']}",
        f"Skills - Frameworks: {row['Skills - Frameworks']}",
        f"Skills - Databases: {row['Skills - Databases']}",
        f"Skills - Tools: {row['Skills - Tools']}",
        f"Skills - Libraries: {row['Skills - Libraries']}",
        f"Skills - Cloud Platforms: {row['Skills - Cloud_platforms']}",
        f"Skills - Soft Skills: {row['Skills - Soft_skills']}",
        f"Skills - Domain Expertise: {row['Skills - Domain_expertise']}"
    ]
    
    return "\n".join(fields)

def generate_multiqueries(client, tools_jd, jd, n):
    prompt = f"""
                    Given the following job description, generate {n} alternative job descriptions using different synonyms and varied phrasing to capture a broader range of keywords for resume matching. 
                    Each alternative should maintain the original responsibilities and be approximately the same length as the original description. Return the output in the specified function format.
              """

    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[
            {"role": "user", "content": prompt + "\n" + jd}
        ],
        temperature=0.1,
        tools=tools_jd,
        tool_choice={"type": "function", "function": {"name": "generate_jd_variants"}}
    )

    jd_dict = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    return [jd_dict['original_jd']] + jd_dict['variant_jds']