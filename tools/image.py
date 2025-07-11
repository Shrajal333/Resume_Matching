import base64

def image_to_base64_tool(image_path):
    with open(image_path, "rb") as img_f:
        return "data:image/jpeg;base64," + base64.b64encode(img_f.read()).decode()
    
def create_multimodal_message_tool(img_paths, prompt):
    messages = [{"role": "system", "content": prompt}]

    for idx, img_path in enumerate(img_paths):
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Page {idx + 1} of the resume:"},
                {"type": "image_url", "image_url": {"url": image_to_base64_tool(img_path)}},
            ]
        })
        
    messages.append({"role": "user", "content": [{"type": "text", "text": "Parse the attached resume (all pages) and return only JSON."}]})
    return messages