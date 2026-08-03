from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA


load_dotenv()
llm=ChatGroq(model='llama-3.3-70b-versatile')
CUSTOM_PROMPT_TEMPLATE="""
use the piecse of information provided in thr correct context to answer  users question .if you dont
know the answer ,just say that you dont know the answer ,dont try to make up any answer.Dont provide anything out of 
thr given context
 Context :{context}
 Question :{question}

 start the answer directly,no small talk  please.
"""
def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(
        template=custom_prompt_template,input_variables=["context","question"]
    )
    return prompt

#load database
DB_FAISS_PATH="vectorestore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
import os
import streamlit as st

st.write("Current working directory:", os.getcwd())

for root, dirs, files in os.walk("."):
    st.write(root, files)
db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)

qa_chain=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}

)

user_query=input('write a query here : ')

response=qa_chain.invoke({'query':user_query})
print(response['result'])



