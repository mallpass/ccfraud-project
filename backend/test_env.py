from dotenv import load_dotenv
import os
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

print("Loaded:", os.getenv("DATABASE_URL"))
