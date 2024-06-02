# RAG for Cancer Webpage:
# - Extracts, chunks, and loads webpage content on cancer into a FAISS database.
# - Uses a "create_history_aware_retriever" chain for a sample conversation.
# - Passes chat history to ask a follow-up question.

import openai
import langchain
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings

# Load documents from a webpage.
loader = WebBaseLoader("https://www.cancer.net/")
docs = loader.load()

# Split large text into chunks based on a specified chunk size.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=30)
documents = text_splitter.split_documents(docs)

# Set API key for OpenAI.
os.environ["OPEN_API_KEY"] = "XXXXX"
llm = ChatOpenAI(openai_api_key = os.environ["OPEN_API_KEY"])
embeddings = OpenAIEmbeddings(openai_api_key = os.environ["OPEN_API_KEY"])

# Store embeddings of documents and perform similarity searches on them using FAISS.
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# Create a prompt for the LLM to generate the search query.
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])

# LLM generates a search query for the retriever by using chat_history.
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

sample_answer = """Briefly intruduce the types of cancer and approciate treatments:
    1. Define and identify the types of cancers.
    2. Explore the advantages and disadvatages cancer treatment methods.
    3. Choose the optimal treatment methods after comparision.
    4. Summarize the findinds.
    ."""

chat_history = [HumanMessage(content="How to early screen out and efficiently prevent cancers?"), 
                AIMessage(content=sample_answer)]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

output = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Can you briefly introduce the types of blood cancers? how to early prevent lymph cancer in children? And elaborate on the optimal methods against lymph cancer?"
})

print(output["answer"])

Output:

Types of Blood Cancers:
1. Leukemia: A cancer of the blood or bone marrow.
2. Lymphoma: A cancer that affects the lymphatic system.
3. Myeloma: A cancer that develops in plasma cells in the bone marrow.

Prevention of Lymphoma in Children:
1. Encouraging a healthy lifestyle with balanced nutrition and regular exercise.
2. Avoiding exposure to known carcinogens, such as tobacco smoke.
3. Ensuring regular check-ups and vaccinations.

Optimal Methods Against Lymphoma:
1. Chemotherapy: Using drugs to kill cancer cells.
2. Radiation therapy: Using high-energy rays to target and destroy cancer cells.
3. Immunotherapy: Boosting the body's immune system to fight cancer.
4. Stem cell transplant: Replacing damaged bone marrow with healthy stem cells.

Early detection through regular screenings and prompt treatment are crucial in effectively combating lymphoma in children.

## Acknowledge:
1). Youtube. 
2). Udemy.
