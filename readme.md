---
title: Ask Medchatbot By Rick 🧑‍⚕️
emoji: 🧑‍⚕️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
short_description: A medical RAG chatbot powered by Llama-3.3-70b and FAISS vectorstore.
env:
  GROQ_API_KEY: your-groq-api-key-here
---

# Ask Medchatbot By Rick 🧑‍⚕️

I built this retrieval-augmented generation (RAG) medical chatbot to answer queries strictly using information extracted from local PDF documents. It utilizes a Streamlit frontend interface, LangChain for handling execution pipelines, Groq (Llama-3.3-70b-versatile) as the core model engine, and a FAISS database for vector search.

---

## 🛠️ Architecture Overview

1. **Ingestion Engine:** I load PDF files from the local directory, split the texts into chunks of 500 characters with a 50-character overlap, and map them using the `all-MiniLM-L6-v2` embedding model.
2. **Retrieval Pipeline:** I configured a `RetrievalQA` pipeline to extract the top 3 most relevant context pieces from the database to answer user inputs.
3. **Guardrails:** I embedded a strict custom prompt constraint instructing the LLM to skip casual small talk and answer using only the provided context. If the answer is unknown, it will state so directly without generating false data.

---

## 📂 Project Structure

I structured this repository with the following files:
* `app.py`: The production web interface built with Streamlit.
* `ingestion.py`: The internal processing script I use to chunk documents and generate the vector store.
* `test_cli.py`: A basic terminal testing script I use to query the database manually via standard input.
* `data/`: The directory where I drop raw medical reference PDFs.
* `vectorestore/db_faiss/`: The local directory where I persist the generated vector store indices.
* `chat_history.json`: The storage path I create dynamically when a user clicks the option to save active sessions.

---

## 🚀 Local Deployment Steps

### 1. Configure the Secrets
I read credentials via an environment file. I create a `.env` file in the root root folder mapping my access token:
```env
GROQ_API_KEY=your_actual_groq_api_key
```

### 2. Install Packages
I install the required application baseline packages using pip:
```bash
pip install streamlit langchain-groq langchain-core langchain-huggingface langchain-community faiss-cpu pypdf dotenv
```

### 3. Build the Database
I place my medical texts inside the `data/` folder, then run my processing engine to index the data locally:
```bash
python ingestion.py
```

### 4. Boot the Web App
I launch the final interactive application frontend locally by spinning up the Streamlit server:
```bash
streamlit run app.py
```

---

## 🛑 Important Infrastructure Notes

* **Ephemereal File Storage:** I designed the chat backup module to save sessions into `chat_history.json`. When running this on cloud servers with ephemeral filesystems, these saved sessions will reset whenever the container sleeps or redeploys.
* **Database Tracking:** I make sure to run the ingestion pipeline locally first and explicitly commit the generated binary vector artifacts inside `vectorestore/` into my remote repository so the cloud service can read them immediately.
