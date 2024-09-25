from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

# Langsmith trackings
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OLLAMA"


# Create ChatpromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant. You will be given a question. You must generate a detailed and long answer."),
        ("user", "Question: {question}"),
    ]
)
@st.cache_data
def generate_response(question, engine, temperature, max_tokens):
    llm = Ollama(model = engine)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Title of the app
st.title("Q&A Chatbot with OLLAMA")

# Sidebar for settings
st.sidebar.title("Settings")
engine = st.sidebar.selectbox("Choose your model", ["llama2", "gemma:2b", "mistral", "phi3"])
                                                    
# Adjust response parameters
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
max_tokens = st.sidebar.number_input("Max tokens", 50, 300, 150)

# Main user input area  
user_input = st.text_input("Ask a question")

if user_input:
    st.text("Your question: " + user_input)
    response = generate_response(user_input, engine, temperature, max_tokens)
    with st.expander("View Answer"):
        st.write(response)

