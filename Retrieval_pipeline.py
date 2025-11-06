import sys
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import re

sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

persistant_dicretory = 'db/chroma_db'
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

db = Chroma(
    persist_directory=persistant_dicretory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


def clean_text(text: str) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def is_medical_report(text: str) -> bool:
    medical_keywords = [
        "hemoglobin", "glucose", "cholesterol", "triglycerides", "hdl", "ldl",
        "rbc", "wbc", "platelet", "urine", "serum", "bilirubin", "creatinine",
        "thyroid", "tsh", "cbc", "mg/dl", "g/dl", "mmol/l", "blood test"
    ]
    found = sum(1 for word in medical_keywords if re.search(rf"\b{word}\b", text.lower()))
    return found >= 2  


def load_and_chunk_user_report(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Loaded {len(documents)} pages, split into {len(chunks)} chunks.")
    return chunks


def explain_uploaded_report(file_path):
    user_chunks = load_and_chunk_user_report(file_path)
    user_text = " ".join([chunk.page_content for chunk in user_chunks])
    user_text = clean_text(user_text)

    # 🚫 Validate if it’s a medical report
    if not is_medical_report(user_text):
        raise ValueError("NOT_MEDICAL_REPORT")

    retriever = db.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(user_text)
    print(f"Retrieved {len(relevant_docs)} relevant chunks from knowledge base.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system_prompt = """
    You are an expert healthcare assistant that analyzes lab reports.
    Your task is to interpret the user's report and provide structured, human-readable explanations.
    Use the retrieved medical knowledge to support your interpretation.

    Format the output exactly as follows for EACH metric/test:
    Name: <test name>
    Actual value: <value from report>
    Normal range: <reference range from knowledge base or report>
    Result: Normal / Low / High / Borderline
    Tips: <suggest simple tips if relevant, e.g., drink water, check with doctor, eat more iron-rich food>

    ⚠️ Important:
    - If the test or its range isn’t found, say "Not specified".
    - Do not include any unrelated commentary.
    - Maintain this exact structured format for every test.
    """

    human_prompt = f"""
    Below is a patient's medical report text extracted from a PDF:

    {user_text}

    Here is additional medical information retrieved from the knowledge base:
    {[clean_text(doc.page_content) for doc in relevant_docs]}

    Please generate a structured report as described in the format above.
    """

    messages = [
        SystemMessage(content=clean_text(system_prompt)),
        HumanMessage(content=clean_text(human_prompt))
    ]

    print("Generating medical explanation...")
    response = llm.invoke(messages)
    print("Explanation generated!\n")

    return clean_text(response.content)


def generate_explanation(file_path):
    return explain_uploaded_report(file_path)
