# This project aims to improve the candidate's resume based on their background and the job description. 
# It will evaluate whether the candidate is a good match for the role and provide suggestions for revision.

# 1). PDF to image (pdf2image)
# 2). Format of pdf_parts (format of jpg, encode to base64)
# 3). Model of Google Gemini Pro
# 4). Prompts instruction
# 5). Streamlit App


import base64
import streamlit as st
import os
import io
from PIL import Image 
import pdf2image
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = "XXXXXXXXXXXXXXXXX"
os.environ["GOOGLE_API_KEY"] = "XXXXXXXXXXXXXXXXXXXXXXXXX"
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(input,pdf_cotent,prompt):
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([input,pdf_content[0],prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        images=pdf2image.convert_from_bytes(uploaded_file.read())
        first_page=images[0]

        img_byte_arr = io.BytesIO()                   
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError ("No file uploaded")

## Streamlit App

st.set_page_config(page_title="Resume Evaluation System")
st.header("Resume Evaluation System")
input_text = st.text_area ("Job Description: ", key="input")
uploaded_file = st.file_uploader ("Upload your resume (PDF)...", type=["pdf"])
# uploaded_file = st.file_uploader ("Upload your resume (docs)...", type=["docs"])


if uploaded_file is not None:
    st.write ("PDF Uploaded Successfully!")


submit1 = st.button ("what is the percentage match? Am I a good fit for this job?")

submit2 = st.button ("How can I best position myself for this job?")

submit3 = st.button ("Please rewrite the resume for me.")

input_prompt1 = """
As an experienced hiring manager in data science, please carefully review the attached candidate resume against the job description. 
Please evaluate whether the candidate's profile aligns with the role.
1. Provide a percentage match based on how well the resume aligns with the job description.
2. Provide a concise summary of your evaluation.
3. Highlight the strengths of the candidate in relation to the specified job requirements.
4. Highlight the weaknesses of the candidate in relation to the specified job requirements.

"""

input_prompt2 = """
As an experienced hiring manager (Senior Data Scientist or Machine Learning Engineer) with a deep understanding of data science:
1. Please carefully review the candidate's resume and the job description. Identify any missing keywords relevant to the role.
2. Provide constructive suggestions to help the candidate improve their resume.
"""

input_prompt3 = """
As a skilled resume writer with professional experience in data science, please carefully rewrite the resume 
based on the attached resume and job description, aiming for a 100 % match for the job description.
Additionally, ensure you incorporate the suggestions from the experienced hiring manager provided above. 
If possible, save the revised resume as a PDF with the name "updated resume" for download.
"""

if submit1:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup (uploaded_file)
        response = get_gemini_response (input_prompt1,pdf_content,input_text)
        st.subheader ("The result is as follows:")
        st.write (response)
    else:
        st.write ("Please upload the resume!")

elif submit2:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup (uploaded_file)
        response = get_gemini_response (input_prompt2,pdf_content,input_text)
        st.subheader ("The Repsonse is")
        st.write (response)
    else:
        st.write ("Please upload the resume!")

elif submit3:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup (uploaded_file)
        response = get_gemini_response (input_prompt3,pdf_content,input_text)
        st.subheader ("The Repsonse is:")
        st.write (response)
    else:
        st.write ("Please upload the resume!")


# Acknowledge:
# 1. Youtube (*)
# 2. Udemy
