import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "./data"
DB_PATH = "./vectorstore/chroma"

def load_documents():
    files = [
        "inventory_report.txt",
        "shipment_schedule.txt",
        "supplier_contracts.txt",
        "demand_forecast.txt"
    ]

    docs = []

    for file in files:
        loader = TextLoader(os.path.join(DATA_PATH, file))
        docs.extend(loader.load())

    return docs


def main():
    print("📥 Loading documents...")

    docs = load_documents()

    print(f"Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    print(f"Split into {len(chunks)} chunks")

    print("🔢 Creating embeddings...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=DB_PATH
    )

    vectorstore.persist()

    print("✅ Vector DB created successfully at:", DB_PATH)


if __name__ == "__main__":
    main()
