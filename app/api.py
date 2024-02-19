from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
import google.generativeai as genai
from system_message import prompt_template
from urls import urls_list
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

# Initialize Google Generative AI embeddings with a specific model and API key
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

# ################################################## URL Knowledge Base #########################################################
loader = AsyncHtmlLoader(urls_list)
docs = loader.load()
bs_transformer = BeautifulSoupTransformer()
tags_to_extract = ['h2', 'p']
doc = bs_transformer.transform_documents(docs, tags_to_extract=tags_to_extract)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits1 = text_splitter.split_documents(doc)
################################################## URL Knowledge Base ################################################################


################################################# CSV Knowledge Base #########################################################
loader = CSVLoader(file_path='QAs.csv')
docs = loader.load()
# Split the documents into chunks of text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits2 = text_splitter.split_documents(docs)
################################################## CSV Knowledge Base #########################################################

# Combine the two splits from URL knowledge Base and CSV file
splits = splits1 + splits2

# Create a Chroma vector store from the document splits using the embeddings
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=gemini_embeddings,
   
)

# Retrieve from vectorstore
retriever = vectordb.as_retriever(search_kwargs={"k": 100000})

# Define the prompt template for generating questions
_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a 
standalone question without changing the content in the given question.

Chat History:
{chat_history}
Follow-Up Input: {question}
Standalone Question:"""
condense_question_prompt_template = PromptTemplate.from_template(_template)

qa_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question", "chat_history"]
)

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY, convert_system_message_to_human=True)

memory = ConversationSummaryMemory(
    memory_key="chat_history",
    return_messages=True,
    llm=llm,
)

question_generator = LLMChain(llm=llm, prompt=condense_question_prompt_template, memory=memory)
doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)
qa_chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    memory=memory,
)


class Query(BaseModel):
    query: str

chat_history = []

@app.get('/')
def index():
    return {"message": "Welcome to the QA Service"}



@app.post('/query')
async def process_query(query: Query):
    data = query.query
    question = data
    if not question:
        raise HTTPException(status_code=400, detail="No query provided")
    try:
        result = qa_chain.invoke({'question': question, 'chat_history': chat_history})
        response = result
        chat_history.append((question, response))
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        response_text = model.start_chat().send_message(f"Paraphrase this {str(response['answer'])} in one sentence and do not lose the context").text
        return JSONResponse(content={'response': response_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
