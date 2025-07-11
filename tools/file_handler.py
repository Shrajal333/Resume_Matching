import os
import shutil
import zipfile
import tempfile
import requests
import streamlit as st
from pathlib import Path
import concurrent.futures
from tools.time import time_tool
from urllib.parse import urlparse
from typing import List, Optional
from tools.model import client_tool
from parsing.resume_formatting import resume_process

class FileHandlerProcessor:
    def __init__(self):
        self.output_dir = Path("temp_resumes")
        
    def download_from_url(self, url: str, progress_bar, status_text) -> Optional[str]:
        # Download a PDF from URL with progress tracking

        try:
            status_text.text(f"Downloading from: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Extract filename from URL
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename or not filename.endswith('.pdf'):
                return None, url
            
            filepath = self.output_dir / filename
            counter = 1
            while filepath.exists():
                name, ext = os.path.splitext(filename)
                filepath = self.output_dir / f"{name}_{counter}{ext}"
                counter += 1

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
            
            return str(filepath), None
            
        except Exception as e:
            st.error(f"Error downloading {url}: {e}")
            return None, url
    
    def extract_gdrive_file_id(self, gdrive_url: str) -> Optional[str]:
        # Extract file ID from Google Drive URL

        if '/file/d/' in gdrive_url:
            return gdrive_url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in gdrive_url:
            return gdrive_url.split('id=')[1].split('&')[0]
        elif '/open?id=' in gdrive_url:
            return gdrive_url.split('/open?id=')[1].split('&')[0]
        return None
    
    def download_from_gdrive(self, gdrive_url: str, progress_bar, status_text) -> Optional[str]:
        # Download from Google Drive

        file_id = self.extract_gdrive_file_id(gdrive_url)
        if not file_id:
            st.error(f"Could not extract file ID from: {gdrive_url}")
            return None, gdrive_url
            
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        return self.download_from_url(download_url, progress_bar, status_text)
    
    def process_uploaded_files(self, uploaded_files: List) -> List[str]:
        # Process files uploaded through Streamlit

        saved_files = []
        ignored_files = []

        for uploaded_file in uploaded_files:
            if uploaded_file.name.lower().endswith('.pdf'):
                filepath = self.output_dir / uploaded_file.name
                
                # Handle duplicates
                counter = 1
                while filepath.exists():
                    name, ext = os.path.splitext(uploaded_file.name)
                    filepath = self.output_dir / f"{name}_{counter}{ext}"
                    counter += 1
                
                with open(filepath, 'wb') as f:
                    f.write(uploaded_file.read())
                saved_files.append(str(filepath))
            
            else:
                ignored_files.append(uploaded_file.name)
        
        return saved_files, ignored_files
    
    def process_zip_file(self, zip_file) -> List[str]:
        # Extract PDFs from uploaded zip file

        extracted_files = []
        ignored_files = []

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                zip_path = os.path.join(temp_dir, "uploaded.zip")
                with open(zip_path, 'wb') as f:
                    f.write(zip_file.read())
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find all PDF files
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith('.pdf'):
                            src_path = os.path.join(root, file)
                            dst_path = self.output_dir / file
                            
                            # Handle duplicates
                            counter = 1
                            while dst_path.exists():
                                name, ext = os.path.splitext(file)
                                dst_path = self.output_dir / f"{name}_{counter}{ext}"
                                counter += 1
                            
                            shutil.copy2(src_path, dst_path)
                            extracted_files.append(str(dst_path))
                        else:
                            ignored_files.append(file)
                            
            except Exception as e:
                st.error(f"Error processing zip file: {e}")
                
        return extracted_files, ignored_files
    
    def process_resumes_parallel(self, filepaths: List[str], max_workers: int = 8) -> List[dict]:
        # Process resumes in parallel with progress tracking

        valid_filepaths = [fp for fp in filepaths if fp.endswith('.pdf')]
        results = []

        if not valid_filepaths:
            return results
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(resume_process, filepath, time_tool(), client_tool()) for filepath in valid_filepaths]
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    flat_data = future.result()
                    results.append(flat_data)
                    completed += 1
                    progress = completed / len(futures)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {completed}/{len(futures)} resumes")

                except Exception as e:
                    st.error(f"Error processing resume: {e}")
                    completed += 1
        
        status_text.text("Processing complete!")
        return results
    
    def cleanup_temp_files(self):
        # Clean up temporary files

        try:
            shutil.rmtree(self.output_dir)
        except Exception as e:
            st.warning(f"Could not clean up temporary files: {e}")