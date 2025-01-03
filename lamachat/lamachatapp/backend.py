from langchain import hub
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
#from langchain.llms import Ollama
#from langchain.embeddings.ollama import OllamaEmbeddings
#from langchain.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
import time

persist_directory = '/tmp/data'
localmodel = "llama3.2"
embeding = "nomic-embed-text"

def pdflrn(file):
    loader = PyPDFLoader(file)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)

    os.system("rm -r /tmp/data")

    vectorstore = Chroma.from_documents(all_splits, embedding=OllamaEmbeddings(model=embeding), persist_directory=persist_directory)
    
    return "OK"

def pdfres(query):
  vectorstore = Chroma(embedding_function=OllamaEmbeddings(model=embeding), persist_directory=persist_directory)
  llm = Ollama(base_url="http://localhost:11434", model=localmodel, verbose=False,
              callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
  retriever = vectorstore.as_retriever()

  template = """ Answer the question based only on the following
  context: {context}

  User: {question}
  Chatbot:
  """
  prompt = PromptTemplate(input_variables=["context","question"], template=template,)
  #memory = ConversationBufferMemory(memory_key="history", return_messages=True, input_key="question")

  qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, verbose=False,
                                        chain_type_kwargs={
                                            #"verbose": True,
                                            "prompt": prompt,
                                            #"memory": memory,
                                        })
  res = qa_chain.invoke({"query": query})
  return res

