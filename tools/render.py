import colorsys
from math import ceil
import streamlit as st

def score_to_color(score):
    hue = (score / 100) * 0.33
    r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

def render_candidate(meta, score):
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"""
        <div style="font-size:1.1em; line-height:1.7;">
            <b>{meta.get('Name','')}</b><br>
            <i>{meta.get('Job Title','')}</i><br>
            <span>Email:</span> {meta.get('Email','')}<br>
            <span>Phone:</span> {meta.get('Phone','')}<br>
            <span>Experience:</span> {meta.get('Experience','')}<br>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

    with col2:
        percentage = ceil(score)
        color = score_to_color(score)
        st.markdown(f"""
        <div style="
            width:100px;height:100px;
            border-radius:50%;
            border:8px solid {color};
            display:flex;align-items:center;justify-content:center;
            font-size:1.7em;font-weight:bold;
            background: linear-gradient(135deg, {color} 0%, #e5e7eb 100%);
            color: #222; margin:auto;">
            {percentage}%
        </div>
        <div style="text-align:center; color:gray; font-size:0.9em; margin-top:0.5em;">
            Match
        </div>
        """, unsafe_allow_html=True)

        resume_path = "temp_resumes/" + meta.get('resume_path','')
        if resume_path:
            with open(resume_path, "rb") as f:
                file_bytes = f.read()
            st.download_button(
                label="📄 Download/View Resume",
                data=file_bytes,
                file_name=resume_path.split("/")[-1],
                mime="application/pdf",
                use_container_width=True,
                key=f"download_{resume_path}"
            )
        st.markdown("---")