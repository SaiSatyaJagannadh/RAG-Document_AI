from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from pathlib import Path
from pypdf import PdfReader

load_dotenv()




big_string = "Rabit company revenue in 2025 was 42,000"

system_prompt = f"""
You are a helpful assistant that answers questions about these documents: {big_string}
"""


documents=[]
# Documents and system prompt
directory = Path("My Documents")
#iterating each file 
for file_path in directory.rglob("*"):
    if file_path.suffix in {".txt", ".pdf", ".md"}:
        #if it si a pdf we use pypdf one 
        if file_path.suffix == ".pdf":
            reader = PdfReader(file_path)
            content = "\n".join(page.extract_text() for page in reader.pages)
            #print(content)
        else:
            content = file_path.read_text()
        documents.append({'path':str(file_path),'content':content})
print(documents)

messages = [
    SystemMessage(content=system_prompt)
]

user_query = input("Enter your query: ")
messages.append(HumanMessage(content=user_query))

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
response = llm.invoke(messages)

print(response.content)