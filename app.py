import os
import pandas as pd
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import initialize_agent, AgentType, tool
from langchain_groq import ChatGroq

# --- Set page config ---
st.set_page_config(page_title="Natural Language to SQL Generator", layout="wide")
st.title("Natural Language to SQL Generator")
st.markdown("Upload your **CSV or TXT** schema file, then ask a question about your data.")

# --- File upload ---
uploaded_file = st.file_uploader("Upload a CSV or TXT schema file", type=["csv", "txt"])
if not uploaded_file:
    st.info("Please upload a file to continue.")
    st.stop()

# --- Load schema based on file type ---
@st.cache_data
def load_schema(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".txt"):
        df = pd.read_csv(file, sep=",")
    else:
        raise ValueError("Unsupported file format. Please upload CSV or TXT.")
    
    df.fillna("", inplace=True)
    
    column_texts = []
    for _, row in df.iterrows():
        table = row['table_name']
        column = row['column_name']
        dtype = row['column_data_type']
        key_type = row['key_type'].strip()
        related = row['related_table'].strip()

        if key_type:
            if related:
                column_text = f"{table}.{column} {dtype} {key_type} KEY {related}"
            else:
                column_text = f"{table}.{column} {dtype} {key_type} KEY"
        else:
            column_text = f"{table}.{column} {dtype}"
        column_texts.append(column_text)

    schema_docs = df.apply(
        lambda row: f"Table `{row['table_name']}` has column `{row['column_name']}` of type `{row['column_data_type']}`. Description: {row['column_description']}. Key Type: {row['key_type'] or 'None'}. Related Table: {row['related_table'] or 'None'}.",
        axis=1
    ).tolist()

    return column_texts, schema_docs

# --- Vectorstore & Agent setup ---
@st.cache_resource
def init_vectorstore_and_agent(schema_docs):
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_texts(schema_docs, embedding_model)
    retriever = lambda q: vectorstore.similarity_search(q, k=2)

    groq_api_key ="<use groq api key>"
    if not groq_api_key:
        st.error("GROQ_API_KEY environment variable not set!")
        st.stop()

    llm = ChatGroq(temperature=0.9, model_name="llama3-8b-8192", groq_api_key=groq_api_key)

    @tool
    def generate_sql(nl_input: str) -> str:
        """Generate SQL from natural language input using the database schema."""
        schema = retriever(nl_input)
        schema_text = "\n".join([doc.page_content for doc in schema])
        prompt = f"-- Given the following schema:\n{schema_text}\n-- Translate the user request into SQL:\n-- Request: {nl_input}\n-- SQL:"
        result = llm.invoke(prompt)
        text = result.content
        return text.strip().split("-- SQL:")[-1].strip()

    agent = initialize_agent(
        tools=[generate_sql],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
    )
    return agent

# --- Load & show schema ---
try:
    column_texts, schema_docs = load_schema(uploaded_file)
except Exception as e:
    st.error(f"Failed to load schema: {e}")
    st.stop()

with st.expander("📋 Sample schema preview"):
    st.write("\n".join(column_texts[:10]))

# --- Initialize agent ---
with st.spinner("Initializing agent..."):
    agent = init_vectorstore_and_agent(schema_docs)

# --- Input box for user query ---
user_input = st.text_area("💬 Ask a question about your data:")

if st.button("Generate SQL"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating SQL..."):
            try:
                sql = agent.run(user_input)
                st.code(sql, language="sql")
            except Exception as e:
                st.error(f"Error generating SQL: {e}")
