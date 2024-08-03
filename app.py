from pandasai.llm.local_llm import LocalLLM
import streamlit as st
import pandas as pd
from pandasai import Agent

#setup for local AI model of choice
model = LocalLLM(
    api_base="http://localhost:11434/v1",
    model="llama3.1"
)

#Page title
st.title("Data Analysis Agent with PandasAI & Llama3.1 8.0b")

#File upload function
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)


if uploaded_file is not None:
    #read uploaded file and show the first 3 rows
    data = pd.read_csv(uploaded_file)
    st.write(data.head(3)) 

    #introduce pandasai Agent to the data and llm
    agent = Agent(data, config={"llm": model})
    prompt = st.text_input("Enter your prompt:")

    #Create generate button and convert user input into data query
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                st.write(agent.chat(prompt))