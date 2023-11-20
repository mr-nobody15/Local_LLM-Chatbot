from langchain .text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS


Data_path="./Documents/"
DB_faiss_path="vector/db_faiss"

#create vector database
def create_vector_database():
    #file load and text_splitter
    print("Document Ingestion has begun...")
    loader=DirectoryLoader(Data_path,glob='*pdf',loader_cls=PyPDFLoader)
    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    texts=text_splitter.split_documents(documents)
    
    # embeddings
    embeddings = HuggingFaceBgeEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',model_kwargs = {'device': 'cpu' })
    db=FAISS.from_documents(texts,embeddings)

    db.save_local(DB_faiss_path)
    print("Ingestion successfull :)")

if __name__=='__main__':
    create_vector_database()

