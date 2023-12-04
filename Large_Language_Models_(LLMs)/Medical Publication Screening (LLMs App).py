## This Project is designed for assisting in screening Medical Publications based on  
## specific descriptions using Large Language Models (LLMs).

import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone      ## Vector databases
from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
import pinecone
from pypdf import PdfReader                      ## Read pdf papers
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain import HuggingFaceHub

import streamlit as st                           ## shareable web apps
from dotenv import load_dotenv
import uuid

## Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

## Input files 
def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:        
        chunks=get_pdf_text(filename)

        ## Adding data and metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"id":filename.id,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))
    return docs

## Embeddings instance
def create_embeddings_load_data():
    embeddings = OpenAIEmbeddings()           ## option
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

## Vector Databases of Pinecone
def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):

    pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )
    print("done......2")
    Pinecone.from_documents(docs, embeddings, index_name=pinecone_index_name)

## Extract infrmation from Vector databases (Pinecone)
def pull_from_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings):

    pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )
    index_name = pinecone_index_name

    index = Pinecone.from_existing_index(index_name, embeddings)
    return index


# Similarity searching 
def similar_docs(query,k,pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,unique_id):

    pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name

    index = pull_from_pinecone(pinecone_apikey,pinecone_environment,index_name,embeddings)
    similar_docs = index.similarity_search_with_score(query, int(k),{"unique_id":unique_id})
    print(similar_docs)                 
    return similar_docs


## Summarization of document
def get_summary(current_doc):
    llm = OpenAI(temperature=0)
    llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})   # option (comparison)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])
    return summary

## Session variables
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] =''

def main():
    load_dotenv()

    st.set_page_config(page_title="Medical Publications Screening Assistance")
    st.title("Medical Publications Screening Assistance...")
    st.subheader("Help you in mdedical publications screening process")

    job_description = st.text_area("Please paste the 'papers sceening' here...",key="1")
    document_count = st.text_input("No.of 'Medical Publications' to return", key="2")
    # Upload the Medical Publications (pdf files)
    pdf = st.file_uploader("Upload medical publications here, only PDF files allowed", type=["pdf"],accept_multiple_files=True)
    submit=st.button("Help me with the analysis")

    if submit:
        with st.spinner('Wait for it...'):
            st.session_state['unique_id']=uuid.uuid4().hex           ## Creating a unique ID
            final_docs_list=create_docs(pdf,st.session_state['unique_id'])
            st.write("*Medical Publications uploaded* :"+str(len(final_docs_list)))

            embeddings=create_embeddings_load_data()                 ## Create embeddings instance

            push_to_pinecone("XXXXXXX","YYYYYYYYY","test",embeddings,final_docs_list)
            relavant_docs=similar_docs(job_description,document_count,"XXXXXXXXX","YYYYYYYYYY","test",embeddings,st.session_state['unique_id'])

            st.write(relavant_docs)
            st.write(":heavy_minus_sign:" * 30)

            for item in range(len(relavant_docs)):               
                st.subheader("Here is "+str(item+1))
                st.write("**File** : "+relavant_docs[item][0].metadata['name'])

                with st.expander('Show me'): 
                    st.info("**Match Score** : "+str(relavant_docs[item][1]))
                    st.write("***"+relavant_docs[item][0].page_content)        ## optional
                    summary = get_summary(relavant_docs[item][0])
                    st.write("**Summary** : "+summary)

        st.success("Thanks! Done!")

if __name__ == '__main__':
    main()



## Acknowledge:
## 1). Udemy courses of LLMs
## 2). LangChain function and library
## 3). OpenAI
## 4). Pinecore vector databases
## 5). Streamlit 
