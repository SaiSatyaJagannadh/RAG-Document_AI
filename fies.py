import os
from pathlib import Path

print("Current working directory:", os.getcwd())
print("This file:", Path(__file__).resolve())
print("TAVILY_API_KEY:", os.getenv("TAVILY_API_KEY"))