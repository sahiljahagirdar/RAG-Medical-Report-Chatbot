import os
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")


def load_documents(docs_path = "docs"):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f'The directory {docs_path} does not exist. Please create it and add PDF files.')
    
    loader = DirectoryLoader(
        path = docs_path,
        glob = "*.pdf",
        loader_cls=PyPDFLoader
    )
    
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f'⚠️ No .pdf files found in {docs_path}. Please add your documents.')

    print(f'Successfully loaded {len(documents)} document(s) from {docs_path}')
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):

    print("🔪 Chunking documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks from {len(documents)} original documents.")
    return chunks


def create_vector_store(chunks,persist_directory='db/chroma_db'):
    """ create and persist ChromeDB vector store """
    print('Creating embeddings and storing in ChromaDB')

    embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

    # Create ChromeDB vector Store
    print('--creating vector DB--')
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = persist_directory,
        collection_metadata = {'hnsw:space':'cosine'}
    )
    print('--- Finished creating vector store ---')

    print(f"vector store created and saved to {persist_directory}")
    return vectorstore




def main():
    docs_path = 'docs'
    documents = load_documents(docs_path)
    chunks = chunk_documents(documents)
    vectorstore = create_vector_store(chunks)

    print("\n Knowledge Base successfully created and stored.")


if __name__ == '__main__':
    main()