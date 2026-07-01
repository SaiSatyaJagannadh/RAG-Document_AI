import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

# Load .env
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Uncomment these to verify your keys are loading
# print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
# print("TAVILY_API_KEY:", os.getenv("TAVILY_API_KEY"))

queries = [
    "coffee health benefits",
    "organic coffee benefits studies",
    "coffee antioxidants wellness",
]


def search_for_articles(queries):
    tavily_search = TavilySearch(
        max_results=5,
        topic="news",
        search_depth="advanced",
        time_range="week",
        include_raw_content=False,
        include_answer=False,
    )

    all_results = []

    for query in queries:
        results = tavily_search.invoke(query)
        all_results.append(results)

    return all_results


def generate_newsletter(search_results):
    prompt = f"""
You are a professional newsletter writer for an organic coffee business.

Below are the search results for this week's coffee news.

SEARCH RESULTS:
{search_results}

Write the article in markdown format.

Rules:
- Focus on positive news about coffee.
- Use headings and bullet points where appropriate.
- Include a short introduction.
- Include a conclusion.
- At the end, provide references using the URLs from the search results.
"""

    model = ChatOpenAI(
        model="gpt-4.1-mini",   # You can also use "gpt-4.1"
        temperature=0.5,
    )

    response = model.invoke(prompt)
    return response.content


if __name__ == "__main__":
    search_results = search_for_articles(queries)
    article = generate_newsletter(search_results)

    print(article)

    with open("article.md", "w", encoding="utf-8") as file:
        file.write(article)

    print("\n✅ Newsletter saved to article.md")