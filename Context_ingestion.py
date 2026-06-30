from pathlib import Path
from pypdf import PdfReader

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load the API key
load_dotenv()


def load_documents(directory="My Documents"):
    documents = []

    # Iterate over each file
    for file_path in Path(directory).rglob("*"):
        if file_path.suffix in {".txt", ".pdf", ".md"}:

            if file_path.suffix == ".pdf":
                reader = PdfReader(file_path)
                content = "\n".join(
                                        page.extract_text() or ""
                                        for page in reader.pages
                                    )
            else:
                content = file_path.read_text()

            documents.append({
                "path": str(file_path),
                "content": content
            })

    return documents


def create_context(document_list):
    context = ""

    for doc in document_list:
        context = context + (
            f"{doc['path']}:\n"
            f"{doc['content']}\n"
            "--------------------\n"
        )

    return context


documents = load_documents()
context = create_context(documents)

system_prompt = f"""
You are a helpful assistant that answers questions about these documents:

{context}
"""

# Set up messages for the LLM
# messages = [
#     SystemMessage(content=system_prompt)
# ]

# user_query = input("Enter your query: ")
# messages.append(HumanMessage(content=user_query))

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# response = llm.invoke(messages)

# print(response.content)

#for multi comm
# Set up conversation
messages = [
    SystemMessage(content=system_prompt)
]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print("Document Chatbot")
print("Type 'exit', 'quit', or 'bye' to end.\n")

while True:
    user_query = input("You: ").strip()

    if user_query.lower() in {"exit", "quit", "bye"}:
        print("Goodbye!")
        break

    # Add user message
    messages.append(HumanMessage(content=user_query))

    # Get response
    response = llm.invoke(messages)

    # Print response
    print(f"\nAssistant: {response.content}\n")

    # Save assistant response so it remembers previous conversation
    messages.append(response)