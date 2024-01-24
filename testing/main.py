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

def init():
    db_embeddings = embedding_functions.OpenAIEmbeddingFunction(
                model_name="text-embedding-ada-002"
            )
    query_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    print("Initialized open api llm and its embeddings")
    # Define a dictionary to map file extensions to their respective loaders
    loaders = {
        '.txt': TextLoader    
    }

    # Define a function to create a DirectoryLoader for a specific file type
    def create_directory_loader(file_type, directory_path):
        return DirectoryLoader(
            path=directory_path,
            glob=f"**/*{file_type}",
            loader_cls=loaders[file_type],
        )

    txt_loader = create_directory_loader('.txt', 'txt')

    # Load the files
    txt_documents = txt_loader.load()

    documents = []
    for i in range(0,len(txt_documents)):
        documents = documents + [txt_documents[i].page_content]
    eppo_source = [os.path.basename(txt_documents[i].metadata['source']).split('_') for i in range(len(txt_documents))]
    metadata = [{'eppo': eppo, 'source': source} for eppo, source in eppo_source]
    ids = [f"id{i}" for i in range(len(documents))]

    #Create ChromaDB collection
    client = chromadb.PersistentClient(path="chromaDB")

    collection = client.get_or_create_collection(name="test",
                                                embedding_function=db_embeddings,
                                                metadata={"hnsw:space": "cosine"})
    collection.add(
     documents=documents,
     ids=ids,
     metadatas=metadata
    )
    globals()['db'] = Chroma(persist_directory="chromaDB", embedding_function=query_embeddings
)
    print("Loading All Contexts Completed")

def load_template():
    prompt = '''Given the following context and question, generate an answer based on this context only.
    Try to provide as much text as possible from "response" section from the source document without making any assumptions.
    If the answer is not found in the context, kindly state "I dont know".
    Dont try to make up an answer.
    Also try to answer in the same language the question is asked.

    CONTEXT:{context}

    QUESTION:{question}
    '''
    bot_prompt = PromptTemplate(
        template=prompt, input_variables=["context","question"]
    )
    print("Loaded Prompt Template")
    return bot_prompt


def retrieval_qa():
    bot_prompt = load_template()
    llm = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        temperature=0.7,
        max_tokens=2000,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=globals()['db'].as_retriever(),
        chain_type_kwargs={"prompt": bot_prompt}
    )