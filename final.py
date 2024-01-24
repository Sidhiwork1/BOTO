import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
import chromadb.utils.embedding_functions as embedding_functions
from langchain_community.chat_models import ChatOpenAI
import chromadb
from langchain.chains import RetrievalQA
from langchain_community.llms.openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Initialize components and models only once
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
db = Chroma(persist_directory="chromaDB", embedding_function=embeddings, collection_name='test')
retriever_openai = db.as_retriever(search_kwargs={'k': 1})


prompt = '''Given the following context and question, generate an answer based on this context only.
Try to provide as much text as possible from "response" section from the source document. 
If the answer is not found in the context, kindly state "I dont know" . 
But if the query is simple like "What is <crop>?", <crop> is a placeholder for any crop, just give them an answer.
If someone greets, greet them back.
CONTEXT:{context}
QUESTION:{question}
'''

bot_prompt = PromptTemplate(
    template=prompt, input_variables=["context", "question"]
)

llm = ChatOpenAI(
    model_name='gpt-3.5-turbo-16k',
    temperature=0.3,
    max_tokens=1000,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Cache for storing previously generated responses
response_cache = {}

def gptcache(question):
    if question in response_cache:
        return response_cache[question]
    
    response = retrieval_qa(question)
    
    # Cache the response to avoid future API calls for the same question
    response_cache[question] = response
    
    return response

def retrieval_qa(question):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_openai,
        return_source_documents=True,
        chain_type_kwargs={"prompt": bot_prompt}
    )
    answer = qa.invoke(question)
    response = answer['result']
    
    if response.strip().lower() == "i dont know":
        return "I don't have the information you're looking for."
    
    return response

