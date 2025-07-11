import os
import json
import fitz
import time
import streamlit as st
from pathlib import Path
from tools.schema import schema_tool
from pdf2image import convert_from_path
from tools.image import create_multimodal_message_tool

def resume_extract_info(file_path):

    all_links = []
    sorted_text_blocks = []
    doc = fitz.open(file_path)

    for page in doc:
        for link in page.get_links(): # Extract links
            if "uri" in link:
                all_links.append(link["uri"])

        blocks = page.get_text("blocks", sort=True) # Extract and sort text blocks
        sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

        for block in sorted_blocks:
            if block[4].strip():
                sorted_text_blocks.append(block[4].strip())

    resume_text = ""
    if all_links:
        resume_text += f"*Links found in the resume:*\n"
        resume_text += "\n".join(all_links) + "\n\n"
    if sorted_text_blocks:
        resume_text += f"*Text extracted from the resume:*\n"
        resume_text += "\n\n".join(sorted_text_blocks)

    if not blocks: # If resumes are images, extract file paths
        images = convert_from_path(file_path, dpi=300)
        img_paths = []
        base = Path(file_path).stem

        for i, img in enumerate(images):
            img_path = f"{base}_page_{i+1}.png"
            img.save(img_path, "PNG")
            img_paths.append(img_path)
        resume_text = None
        return {"fallback_img_paths": img_paths}
    
    return {"resume_text": resume_text}

def resume_text_2_json(resume_info, current_month_year, client):
    tools = schema_tool()

    prompt = f"""
    You are an intelligent resume parser. From the resume text below, extract and return a JSON object with the following fields. 
    Maintain structure strictly, even if some fields are missing (use null, empty string, or empty array as needed). Use consistent formatting as per the schema expectations:

    - Candidate Name (e.g., "Jane Doe")
    - Candidate Email (e.g., "jane.doe@email.com")
    - Candidate Phone Number (e.g., "+91-1234567890")
    - Job Title: Infer the most suitable target job title based on candidate's skills, experience, and projects (e.g., "Data Analyst", "DevOps Engineer", "Machine Learning Engineer"). This may not be explicitly stated in the resume.
    - Candidate Years of Experience (as of {current_month_year}, e.g., "3.5 years"). Only include professional experience. Do not consider internships or projects.
    - Online Profiles: Linkedin, Github, Portfolio, and Others (array of strings)
    - Education: degree, institution, location, GPA (if available), start_date (e.g., "2019-08"), end_date (e.g., "2023-05")
    - Experience: role, organization, location, start_date, end_date, responsibilities (array of bullet points). Include both professional and internship experience.
    - Projects: title, organization (if any), and a description
    - Certificates: List of certifications
    - Awards: List of recognitions
    - Papers / Publications: title, conference (if applicable), status (e.g., "Published", "Under Review")
    - Skills: categorized into languages, frameworks, databases, tools, libraries, cloud platforms, soft skills and domain expertise

    Respond ONLY with a valid JSON object. No commentary.
    """

    if resume_info.get("resume_text"):
        resume_text = resume_info["resume_text"]

        response = client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": resume_text}
            ],
            temperature=0.1,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "extract_resume_info"}}
        )
    
    elif resume_info.get("fallback_img_paths"):
        img_paths = resume_info["fallback_img_paths"]
        messages = create_multimodal_message_tool(img_paths, prompt)

        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=messages,
            temperature=0.1,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "extract_resume_info"}}
        )

        for img_path in img_paths:
            os.remove(img_path)
    
    return json.loads(response.choices[0].message.tool_calls[0].function.arguments)

def process_resumes(processor, max_workers):
    start_time = time.time()
    results = []
    ignored_files = []

    if st.session_state.last_upload.get('type') == 'file':
        files = st.session_state.last_upload.get('files')

        if files:
            saved_files, ignored = processor.process_uploaded_files(files)
            ignored_files.extend(ignored)

            if saved_files:
                results = processor.process_resumes_parallel(saved_files, max_workers)

    elif st.session_state.last_upload.get('type') == 'url':
        urls_text = st.session_state.last_upload.get('urls_text', "")

        if urls_text.strip():
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            downloaded_files = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, url in enumerate(urls):
                progress_bar.progress((i + 1) / len(urls))

                if 'drive.google.com' in url:
                    filepath, ignored = processor.download_from_gdrive(url, progress_bar, status_text)
                else:
                    filepath = processor.download_from_url(url, progress_bar, status_text)
                    filepath, ignored = processor.download_from_gdrive(url, progress_bar, status_text)

                if filepath:
                    downloaded_files.append(filepath)
                if ignored:
                    ignored_files.append(ignored)

            if downloaded_files:
                results = processor.process_resumes_parallel(downloaded_files, max_workers)

    elif st.session_state.last_upload.get('type') == 'zip':
        zip_file = st.session_state.last_upload.get('zip_file')

        if zip_file:
            extracted_files, ignored = processor.process_zip_file(zip_file)
            ignored_files.extend(ignored)

            if extracted_files:
                results = processor.process_resumes_parallel(extracted_files, max_workers)

    # Show ignored files/links if any
    if ignored_files:
        st.warning("The following files/links were ignored (not PDF or download failed):")
        st.markdown("```text\n" + "\n".join(str(f) for f in ignored_files) + "\n```")

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    st.session_state.processed_data = results
    st.session_state.processing_complete = True
    st.session_state.processing_time = elapsed_time