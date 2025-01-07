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
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
import time

persist_directory = 'data'
localmodel = "llama3.2"
embeding = "nomic-embed-text"

def pdflrn(file):
    loader = PyPDFLoader(file)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)

    from pypdf import PdfReader

    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    #all_splits = text_splitter.split_text(text)
    
    
    #os.system("rm -r /tmp/data")

    vectorstore = Chroma.from_documents(all_splits, embedding=OllamaEmbeddings(model=embeding), persist_directory=persist_directory)
    #vectorstore = Chroma.from_texts(all_splits, embedding=OllamaEmbeddings(model=embeding), persist_directory=persist_directory)
    
    return all_splits

def pdfres(query):
  vectorstore = Chroma(embedding_function=OllamaEmbeddings(model=embeding), persist_directory=persist_directory)
  llm = OllamaLLM(base_url="http://localhost:11434", model=localmodel, verbose=False,
              callbacks=[StreamingStdOutCallbackHandler()])
  retriever = vectorstore.as_retriever()

  template = """
  Enter your question: 

  CONTEXT:
  {context}

  QUESTION:
  {question}

  CHAT HISTORY:
  {chat_history}

  ANSWER:
  """
                  
  prompt = PromptTemplate(input_variables=["chat_history", "question", "context"], template=template,)
  memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question")
  chain_type_kwargs={
        "prompt": prompt,
        "memory": memory}
  qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, verbose=False)

  res = qa_chain.invoke({"query": query})                 
    

  '''
  from langchain.chains.question_answering import load_qa_chain
  template = """
  {Your_Prompt}

  CONTEXT:
  {context}

  QUESTION:
  {query}

  CHAT HISTORY:
  {chat_history}

  ANSWER:
  """

  prompt = PromptTemplate(input_variables=["chat_history", "query", "context"], template=template)

  memory = ConversationBufferMemory(memory_key="chat_history", input_key="query")

  chain = load_qa_chain(llm, chain_type="stuff", retriever=retriever, memory=memory, prompt=prompt)
  '''

    
  
  return res

