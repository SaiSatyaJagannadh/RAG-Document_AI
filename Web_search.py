import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

# Load .env
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def search_for_articles(query):
    tavily_search = TavilySearch(
        max_results=5,
        topic="news",
        search_depth="advanced",
        time_range="week",
        include_raw_content=False,
        include_answer=False,
    )

    results = tavily_search.invoke(query)
    return results


def generate_description(search_results, topic):
    prompt = f"""
You are a professional content writer.

Write a detailed markdown article about:

TOPIC:
{topic}

SEARCH RESULTS:
{search_results}

Rules:
- Use markdown.
- Add a title.
- Add an introduction.
- Use headings.
- Use bullet points where appropriate.
- End with a conclusion.
- Add a References section using the URLs from the search results.
"""

    model = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.5,
    )

    response = model.invoke(prompt)
    return response.content


if __name__ == "__main__":
    topic = input("Enter topic to research: ")

    print("\nSearching...\n")
    search_results = search_for_articles(topic)

    print("Generating article...\n")
    article = generate_description(search_results, topic)

    with open("description.md", "w", encoding="utf-8") as file:
        file.write(article)

    print(article)
    print("\n✅ Saved as description.md")