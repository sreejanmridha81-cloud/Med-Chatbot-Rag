import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import json  

load_dotenv()

DB_FAISS_PATH = "vectorestore\db_faiss"
HISTORY_FILE = "chat_history.json"  

CUSTOM_PROMPT_TEMPLATE = """use the piecse of information provided in thr correct context to answer  users question .if you dont
know the answer ,just say that you dont know the answer ,dont try to make up any answer.Dont provide anything out of 
thr given context
 Context :{context}
 Question :{question}

 start the answer directly,no small talk  please.
"""

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

llm = ChatGroq(model='llama-3.3-70b-versatile')

def main():
    st.title("Ask  Medchatbot By Rick 🧑‍⚕️ !")
    st.sidebar.title("Chat Management")
    if st.sidebar.button("💾 Save Current Chat"):
        if 'messages' in st.session_state and len(st.session_state.messages) > 0:
            with open(HISTORY_FILE, "w") as f:
                json.dump(st.session_state.messages, f)
            st.sidebar.success("Chat history saved successfully!")
        else:
            st.sidebar.warning("No messages to save yet.")
    if st.sidebar.button("📂 Load Previous Chat"):
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                st.session_state.messages = json.load(f)
            st.sidebar.success("Previous chat history loaded!")
            st.rerun()  
        else:
            st.sidebar.error("No saved chat history file found.")
            
    
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])   

    prompt = st.chat_input("pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        try:
            vectore_store = get_vectorstore()
            if vectore_store is None:
                st.error("Failed to load the vector Store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=vectore_store.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
    
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({'query': prompt})   
                Final_result = response["result"]

            st.chat_message('assistant').markdown(Final_result)
            st.session_state.messages.append({'role': "assistant", 'content': Final_result})

        except Exception as e:
            st.error(f"error :{str(e)}")

if __name__ == "__main__":
    main()
