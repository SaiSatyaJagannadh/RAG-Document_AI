from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage

load_dotenv()


# -----------------------------
# Load documents
# -----------------------------
def load_documents(directory="My Documents"):
    docs = []

    for file_path in Path(directory).rglob("*"):

        if file_path.suffix not in {".txt", ".pdf", ".md"}:
            continue

        if file_path.suffix == ".pdf":
            reader = PdfReader(file_path)
            content = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )
        else:
            content = file_path.read_text(encoding="utf-8")

        docs.append(
            Document(
                page_content=content,
                metadata={"source": str(file_path)}
            )
        )

    return docs


documents = load_documents()

print(f"Loaded {len(documents)} documents.")


# -----------------------------
# Split into chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


# -----------------------------
# Create embeddings
# -----------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# -----------------------------
# Create Vector Database
# -----------------------------
vector_db = FAISS.from_documents(
    chunks,
    embeddings
)

print("Vector DB Created!")

vector_db.save_local("faiss_index")

print("Vector DB saved!")

# -----------------------------
# Retriever
# -----------------------------
retriever = vector_db.as_retriever(
    search_kwargs={"k": 4}
)


# -----------------------------
# LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

print("\nRAG Chatbot Started")
print("Type exit to quit.\n")


# -----------------------------
# Chat Loop
# -----------------------------
while True:

    question = input("You: ")

    if question.lower() in {"exit", "quit", "bye"}:
        break

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    prompt = f"""
You are a helpful assistant.

Answer ONLY from the context below.
If the answer is not present, say:
"I couldn't find that information."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    print("\nAssistant:")
    print(response.content)

    print("\nSources:")
    for doc in retrieved_docs:
        print("-", doc.metadata["source"])

    print("-" * 60)