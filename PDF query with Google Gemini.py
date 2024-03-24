# This project implements a question-answering system for PDFs using Google Gemini Pro.
# Upload PDF files.
# Extract text from PDF files and split it into text chunks.
# Utilize Google Gemini Pro model to answer user questions based on the processed PDFs.
# Integrate question embedding and indexing for efficient retrieval.
# Deploy using Streamlit.

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
# from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

# Function to extract text from PDF files and create text chunks
def text_vector(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=10)
    chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorDBs = FAISS.from_texts(chunks, embedding=embeddings)
    vectorDBs.save_local("faiss_index")
    return vectorDBs


# Function to get the conversational chain
def get_conversational_chain():
    prompt_template = """
    In a comprehensive and informative way, answer the following question based on the provided context:
  
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Function to handle user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    # Return the response for use in the main function
    return chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)


def main():
    st.set_page_config("Chat with PDF")

    # Main content area
    st.header("Ready to Chat PDF?")
    st.info("I'm here to answer any questions you have about your PDFs.")

    with st.container():
        user_question = st.text_input("Ask a Question")

        if user_question:
            response = user_input(user_question)  # Call user_input and store response
            st.markdown("## Your Answer:")
            st.write(response["output_text"])

    # Sidebar for file uploads and actions
    with st.sidebar:
        st.header("Upload Your PDFs")
        pdf_docs = st.file_uploader("Drop your PDFs here", accept_multiple_files=True)

        if pdf_docs:
            with st.spinner("Processing..."):
                text_vector(pdf_docs)
                st.success("PDFs processed!")

if __name__ == "__main__":
    os.environ["GOOGLE_API_KEY"] = "XXXXXXXXXXXXXXXXXXXXXXX"  # Replace with your actual Google API key
    main()