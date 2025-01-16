from langchain import hub
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
import time



def pdflrn(file,path,embedmodel):
  loader = PyPDFLoader(file)
  data = loader.load()


  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
  all_splits = text_splitter.split_documents(data)


  vectorstore = Chroma.from_documents(all_splits, embedding=OllamaEmbeddings(model=embedmodel), persist_directory=path)
  return True

def pdfres(query,path,model,embedmodel):
  vectorstore = Chroma(embedding_function=OllamaEmbeddings(model=embedmodel), persist_directory=path)
  llm = OllamaLLM(base_url="http://localhost:11434", model=model, verbose=True,                  
              callbacks=[StreamingStdOutCallbackHandler()])
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
  return res["result"]
  

def chat(query):
  llm = ChatOllama( model="llama3.2", temperature=1 )
  for chunk in llm.stream(query):
    yield chunk.content
