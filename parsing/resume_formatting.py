import os
from collections import defaultdict
from parsing.resume_processing import resume_text_2_json
from parsing.resume_processing import resume_extract_info

def resume_json_2_row(tool_args):

    # Basic information
    flat_data = defaultdict(str)
    flat_data['Name'] = tool_args.get('candidate_name', '')
    flat_data['Email'] = tool_args.get('candidate_email', '')
    flat_data['Phone'] = tool_args.get('candidate_phone', '')
    flat_data['Job Title'] = tool_args.get('job_title', '')
    flat_data['Experience'] = tool_args.get('years_of_experience', '')

    # Profiles
    profiles = tool_args.get('online_profiles') or {}
    for key in ['linkedin', 'github', 'portfolio']:
        flat_data[f'Profile - {key.capitalize()}'] = profiles.get(key, '')
    flat_data['Profile - Others'] = ", ".join(profiles.get('others') or [])

    # Education
    edu_lines = []
    for edu in tool_args.get("education", []):
        edu_lines.append(
            f"{edu.get('degree', '')}, {edu.get('institution', '')}, {edu.get('location', '')}, GPA: {edu.get('gpa', '')}, {edu.get('start_date', '')} - {edu.get('end_date', '')}"
        )
    flat_data['Education'] = " | ".join(edu_lines)

    # Experience
    exp_lines = []
    for exp in tool_args.get("experience", []):
        responsibilities = "; ".join(exp.get("responsibilities", []))
        exp_lines.append(
            f"{exp.get('role', '')} at {exp.get('organization', '')} ({exp.get('start_date', '')} to {exp.get('end_date', '')}): {responsibilities}"
        )
    flat_data['Experience Details'] = " | ".join(exp_lines)

    # Projects
    proj_lines = []
    for proj in tool_args.get("projects", []):
        proj_lines.append(
            f"{proj.get('title', '')} - {proj.get('organization', '')}: {proj.get('description', '')}"
        )
    flat_data['Projects'] = " | ".join(proj_lines)

    # Awards and Certificates
    flat_data['Awards'] = ", ".join(tool_args.get("awards") or [])
    flat_data['Certificates'] = ", ".join(tool_args.get("certificates") or [])

    # Publications
    pub_lines = []
    for pub in tool_args.get("publications", []):
        pub_lines.append(
            f"{pub.get('title', '')} - {pub.get('conference', '')} ({pub.get('status', 'N/A')})"
        )
    flat_data['Publications'] = " | ".join(pub_lines)

    # Skills
    for category, items in (tool_args.get("skills") or {}).items():
        flat_data[f"Skills - {category.capitalize()}"] = ", ".join(items or [])

    return flat_data

def resume_process(filepath, current_month_year, client):
    resume_info = resume_extract_info(filepath)
    tool_args = resume_text_2_json(resume_info, current_month_year, client)
    flat_data = resume_json_2_row(tool_args)
    flat_data['resume_path'] = os.path.basename(filepath)
    return flat_data