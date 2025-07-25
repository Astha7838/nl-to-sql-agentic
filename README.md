# nl-to-sql-agentic
# Natural Language to SQL Generator

This project is a Streamlit-based app that allows users to upload a database schema (in `.csv` or `.txt` format), ask natural language questions, and get SQL queries generated using an LLM-powered agent.

---

## 🚀 Features

- Accepts **CSV** (pre-parsed schema) and **TXT** (raw SQL `CREATE TABLE` statements)
- Generates SQL queries using **ChatGroq** + **LangChain agents**
- Visualizes extracted schema
- Streamlit interface for ease of use
- Handles **foreign key references**, **primary keys**, and more
- Supports BAAI's `bge-small-en-v1.5` embedding model with FAISS vectorstore

---

## 🏗️ Project Structure

├── app.py # Main Streamlit application
├── README.md # This file
├── requirements.txt # Python dependencies
└── data_ai.csv # Example CSV schema file (optional)


---

## 📦 Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/nl-to-sql-agentic.git
cd nl-to-sql-agentic

2. Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Run the app:
streamlit run app.py


How It Works
	1	User uploads a CSV or TXT schema file.
	2	Schema is parsed into column descriptions.
	3	LangChain embeddings + FAISS vectorstore are initialized.
	4	A LangChain agent with tools powered by ChatGroq (llama3-8b-8192) is set up.
	5	User asks a natural language question → SQL is generated!

Note:

Ensure that the uploaded schema matches the expected format.
You must set a valid Groq API Key inside the app or use environment variables.
Parsing errors are handled gracefully.


