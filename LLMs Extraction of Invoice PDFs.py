# This script uses LangChain and OpenAI's GPT-3.5 to extract data from invoice PDFs.
# It processes uploaded PDFs, extracts relevant invoice details, and presents the data in a Streamlit application. 
# The script performs the following tasks:

# 1. Setup and Initialization: It initializes the OpenAI API key and sets up the necessary models and templates.
# 2. PDF Processing: It defines a function to read and extract text from user-uploaded PDF invoices.
# 3. Data Extraction: It uses LangChain's LLMChain to process the extracted text and extract relevant invoice fields.
# 4. Streamlit Application: It provides a user interface for uploading PDF files, extracting data, and downloading the results as a CSV file.
# 5. Task Completion: It confirms the completion of the invoice extraction task to the user.
# Here is the complete code:


import os
import openai
import langchain
from pypdf import PdfReader
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv

# Set OpenAI API key
os.environ["OPEN_API_KEY"] = "XXXXXXXX"

# Initialize the ChatOpenAI model
llm = ChatOpenAI(openai_api_key=os.environ["OPEN_API_KEY"])

# Define the prompt template for the documentation writer
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world-class technical documentation writer."),
    ("user", "{input}")
])

# Define the output parser
output_parser = StrOutputParser()

# Create the chain for generating documentation
# chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)
chain = prompt | llm | output_parser

# Generate output using the chain
documentation_output = chain.invoke({"input": "how can langsmith help with testing?"})
print(documentation_output)

# Redefine and initialize the ChatOpenAI model for extraction
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=os.environ["OPEN_API_KEY"],
    temperature=0,
    max_tokens=2000
)

# Function to create documents from user-uploaded PDF files
def create_docs(user_pdf_list):
    """This function is used to extract invoice data from the given PDF files. 
    It uses the LangChain agent to extract the data from the given PDF files."""
    df = pd.DataFrame({
        'Invoice no.': pd.Series(dtype='str'),
        'Item': pd.Series(dtype='str'),
        'Quantity': pd.Series(dtype='str'),
        'Rate': pd.Series(dtype='str'),
        'Amount': pd.Series(dtype='str'),
        'Date': pd.Series(dtype='str'),
        'Ship Mode': pd.Series(dtype='str'),
        'Balance Due': pd.Series(dtype='str'),
        'Subtotal': pd.Series(dtype='str'),
        'Shipping': pd.Series(dtype='str'),
        'Total': pd.Series(dtype='str'),
        'Notes': pd.Series(dtype='int'),
        'Order ID': pd.Series(dtype='str'),     
        'Bill To': pd.Series(dtype='str'),
        'Ship To': pd.Series(dtype='str')
    })

    for filename in user_pdf_list:
        # Extract PDF Data
        texts = ""
        print("Processing -", filename)
        pdf_reader = PdfReader(filename)
        for page in pdf_reader.pages:
            texts += page.extract_text()

        template = """Extract all the following values: invoice no., Item, Quantity, Rate, date, Amount, Date, Ship Mode, Balance Due, Subtotal, 
        Shipping, Total, Notes, Order ID, Bill to and Ship To the following Invoice content: 
            {texts}
            The fields and values in the above content may be jumbled up as they are extracted from a PDF. Please use your judgment to align
            the fields and values correctly based on the fields asked for in the question above.
            Expected output format: 
            {{'Invoice no.': 'xxxxxxxx','Item': 'xxxxxx','Quantity': 'x','Date': 'mm/dd/yyyy', 'Ship Mode': 'xxx xxxx', 'Balance Due': 'xxx.xx', 
            'Rate': 'xxx.xx','Amount': 'xxx.xx', 'Subtotal': 'xxx.xx', 'Shipping': 'xxx.xx', 'Total': 'xxx.xx',
            'Notes': 'xxxxxxxxx','Order ID': 'xxxxxxxxxx','Bill To': 'xxxxxxxxx', 'Ship To': 'xxx.xx'}}
            Remove any dollar symbols or currency symbols from the extracted values.
            """
        prompt = PromptTemplate.from_template(template)

        extraction_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613", openai_api_key=os.environ["OPEN_API_KEY"])

        chain = LLMChain(llm=llm, prompt=prompt)

        data_dict = chain.run(texts)

        print("Dict:...", data_dict)
        new_row_df = pd.DataFrame([eval(data_dict)], columns=df.columns)
        df = pd.concat([df, new_row_df], ignore_index=True)  

        print("********************Good Job!***************")

    print(df) 
    return df

# Streamlit application for invoice extraction
def main():
    load_dotenv()

    st.set_page_config(page_title="Invoice Extraction ...")
    st.title("Invoice Extraction Project")
    st.subheader("We are extracting invoice data")

    # Upload the Invoices (pdf files)
    pdf = st.file_uploader("Upload invoices (PDF) only", type=["pdf"], accept_multiple_files=True)

    submit = st.button("Extract Data")

    if submit:
        with st.spinner('Wait for output'):
            df = create_docs(pdf)
            st.write(df.head())

            data_as_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download data as CSV", 
                data_as_csv, 
                "new_extraction.csv",
                "text/csv",
                key="download-tools-csv"
            )
        st.success("Task of Invoice Extraction is complete")

# Invoking main function
if __name__ == '__main__':
    main()

# Output:

# Dict:... {
#     'Invoice no.': '16384',
#     'Item': 'Bush Stackable Bookrack, Pine',
#     'Quantity': '7',
#     'Date': '12/08/2012',
#     'Ship Mode': 'Standard Class',
#     'Balance Due': '6208.84',
#     'Rate': '874.02',
#     'Amount': '6118.14',
#     'Subtotal': '6118.14',
#     'Shipping': '90.70',
#     'Total': '6208.84',
#     'Notes': 'Thanks for your business!',
#     'Order ID': 'ES-2012-AH10075139-41251',
#     'Bill To': 'Adam Hart',
#     'Ship To': 'Nottingham, England, United Kingdom'
# }
# ********************Good Job***************

# 0       16384                      Bush Stackable Bookrack, Pine        7  ...  ES-2012-AH10075139-41251  Adam Hart    Nottingham, England, United Kingdom
# 1       30118  Harbour Creations Steel Folding Chair, Set of Two        9  ...    IN-2012-AH100757-41143  Adam Hart  Bunbury, Western Australia, Australia
# 2       30845                               Konica Inkjet, White        4  ...    ID-2012-AH100757-41163  Adam Hart            Melton, Victoria, Australia


# [1 rows x 15 columns]
# Langsmith can help with testing in a variety of ways. Here are some ways in which Langsmith can assist with testing:

# 1. **Automated Testing**: Langsmith can be used to write automated test scripts for various testing scenarios. By leveraging its language processing capabilities, Langsmith can generate test cases based on the requirements and specifications provided.

# 2. **Test Data Generation**: Langsmith can be used to generate test data for different test scenarios. It can create realistic and diverse data sets to ensure comprehensive test coverage.

# 3. **Test Case Management**: Langsmith can help in managing test cases by organizing them, linking them to requirements, and tracking their execution status. It can also assist in generating test reports and documentation.

# 4. **Code Quality Analysis**: Langsmith can analyze code quality metrics, identify potential issues, and suggest improvements. This can help in ensuring that the software under test meets quality standards.

# 5. **Integration Testing**: Langsmith can assist in integration testing by generating test cases that cover interactions between different components or systems. It can help in identifying integration issues early in the development cycle.

# 6. **Regression Testing**: Langsmith can automate regression testing by reusing test cases and data sets for each new release or update. This can help in quickly identifying any regressions introduced in the software.

# 7. **Load and Performance Testing**: Langsmith can help in creating test scripts for load and performance testing to evaluate the system's behavior under different load conditions. It can simulate user interactions and measure system response times.

# 8. **Security Testing**: Langsmith can assist in security testing by generating test cases to identify vulnerabilities in the software. It can help in checking for common security issues such as injection attacks, cross-site scripting, and authentication flaws.

# Overall, Langsmith can be a valuable tool in the testing process by automating repetitive tasks, improving test coverage, and ensuring the quality and reliability of the software being developed.