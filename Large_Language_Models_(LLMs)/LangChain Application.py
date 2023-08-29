## OpenAI API (Streamlit)
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import streamlit as st

os.environ["OPENAI_API_KEY"] = "XXXXX"

def load_answer(question):
    llm = OpenAI(model_name="text-davinci-003",temperature=0)
    answer=llm(question)
    return answer

# App UI 
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

# User input
def get_text():
    input_text = st.text_input("You: ", key="input")   # example: Who is Einstein? 
    return input_text

user_input=get_text()
response = load_answer(user_input)

submit = st.button('Generate')  

# Clicked button
if submit:
    st.subheader("Answer:")
    st.write(response)


# Prompt Templates (1)
first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about {name}"
)

## OPENAI LLMS
llm=OpenAI(temperature=0.8)
chain=LLMChain(llm=llm, prompt=first_input_prompt,verbose=True, output_key="person")


## Prompt Templates (2)
second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="when was {person} born?"
)

chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob')


## Prompt Templates (3)
third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="List Nobel Prize winners {dob} in the world"
)
chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description')
parent_chain=SequentialChain(chains=[chain, chain2, chain3], input_variables=['name'],output_variables=['person','dob', 'description'], verbose=True)


if user_input:
    st.write(parent_chain({'name':user_input}))


# Note: All above is for educational purpose only (basic knowledge of LangChain), some modifications were made (thanks to Krish Naik).

