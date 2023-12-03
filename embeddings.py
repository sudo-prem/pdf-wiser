from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
import os

def recursive_document_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 200,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(data)
    return texts

def recursive_text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 200,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_text(data)
    return texts

def vector_db_exists(vector_db_path):
    return os.path.exists(vector_db_path)

class EmbeddingSearchEngine:
    def __init__(self, model_name=None, model_type=None, storage_location="./chroma_db"):
        if model_type == "SentenceTransformer":
            self.model = HuggingFaceEmbeddings(model_name=model_name)
        else:
            self.model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
        self.vector_db = None
        self.storage_location = storage_location

        if vector_db_exists(storage_location):
            self.vector_db  = Chroma(persist_directory=storage_location, embedding_function=self.model)



    def create_embeddings_for_pdf(self, pdf_file_path):
        loader = PyMuPDFLoader(pdf_file_path)
        data = loader.load()
        texts = recursive_document_splitter(data)
        self.vector_db = Chroma.from_documents(texts, self.model, persist_directory=self.storage_location)
        return self
    
    def create_embeddings(self, text):
        texts = recursive_text_splitter(text)
        self.vector_db = Chroma.from_documents(texts, self.model, persist_directory=self.storage_location)
        return self

    def search_embeddings(self, query):

        if self.vector_db == None:
             print("Vector db does not exists")
             return ""
        docs = self.vector_db.similarity_search(query)
        return docs
    
    def check_if_embeddings_exist(self):
        return vector_db_exists(self.storage_location)


if __name__ == "__main__":
    pdf_file_path = "sodapdf-converted.pdf"

    embedding_engine = EmbeddingSearchEngine(storage_location=pdf_file_path.split(".pdf")[0])
    if not embedding_engine.check_if_embeddings_exist():
        print("Embeddings does not exist, so creating one")
        embedding_engine = embedding_engine.create_embeddings_for_pdf(pdf_file_path=pdf_file_path)
    print(embedding_engine.search_embeddings("What did Clara stumble upon?"))

