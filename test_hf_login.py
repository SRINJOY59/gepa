"""Quick test: verify HF login from .env works."""
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HF_TOKEN")
print(f"Token found: {token[:10]}..." if token else "ERROR: HF_TOKEN not found in .env!")

from huggingface_hub import login, whoami

login(token=token)
info = whoami()
print(f"Logged in as: {info['name']}")
print("HF login is working!")
