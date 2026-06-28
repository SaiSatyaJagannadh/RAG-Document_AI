from pathlib import Path

p = Path("My Documents")
print("Exists:", p.exists())
print("Absolute path:", p.resolve())
print("Files:", list(p.rglob("*")))