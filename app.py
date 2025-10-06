import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from fastapi import FastAPI
from langserve import add_routes
from pydantic import BaseModel

class QueryRequest (BaseModel) :
    query: str

os.environ['OPEN_API_KEY']=os.getenv("OPENAI_API_KEY")
## LangSmith Tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACKING_V2']="true"
os.environ['LANGCHAIN_PROJECT']=os.getenv("LANGCHAIN_PROJECT")

try :
    
    loader = WebBaseLoader('https://en.wikipedia.org/wiki/OpenAI')
    docs = loader.load()
    # Splitting Data
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=550,chunk_overlap=150)
    documents = text_splitter.split_documents(docs)

    documents_page_content = []
    for i in documents :
        documents_page_content.append(i.page_content)

    clean_docs = [item.replace("\n", "").strip() for item in documents_page_content]
    embeddings = OpenAIEmbeddings()
    embeddings_documents = embeddings.embed_documents(clean_docs)
    langchain_document  = [Document(page_content=text) for text in clean_docs]
    vectorstoredb = FAISS.from_documents(langchain_document,embeddings)
    llm = ChatOpenAI(model="gpt-4o") # Calling the specific model

    retriever = vectorstoredb.as_retriever(search_kwargs={"k": 3})
    # Build Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    app = FastAPI(title="LangChain Server",
                  version="1.0",
                  description="Question and Answer about the Open AI History"
                  )
    
    add_routes(
        app,
        qa_chain,
        path='/chain'
    )

    @app.post("/ask")
    def ask_question(request: QueryRequest):
        user_query = request.query
        # Call your QA chain
        result = qa_chain.run(user_query)
        return {"output": result}

    if __name__=="__main__" :
        import uvicorn
        uvicorn.run(app,host="0.0.0.0",port=8000)
except RuntimeError as e:
    print(e)