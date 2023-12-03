from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings


def recursive_splitter(data):
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 200,
            chunk_overlap  = 20,
            length_function = len,
            is_separator_regex = False,
        )
        texts = text_splitter.split_documents(data)
        return texts

class EmbeddingSearchEngine:
    def __init__(self, model_name='paraphrase-distilroberta-base-v1', model_type="SentenceTransformer"):
        if model_type == "SentenceTransformer":
            self.model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        else:
            self.model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
        self.vector_db = None
        self.storage_location = None

    def create_embeddings_for_pdf(self, pdf_file):
        loader = PyMuPDFLoader(pdf_file)
        data = loader.load()
        texts = recursive_splitter(data)
        self.vector_db = Chroma.from_documents(texts, self.model)
        return self

    def search_embeddings(self, query):
        docs = self.vector_db.similarity_search(query)
        return docs


if __name__ == "__main__":
    embedding_engine = EmbeddingSearchEngine()
    pdf_file = "sodapdf-converted.pdf"
    embedding_engine = embedding_engine.create_embeddings_for_pdf(pdf_file=pdf_file)
    print(embedding_engine.search_embeddings("What did Clara stumble upon?"))

