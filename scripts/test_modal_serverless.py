import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
modal_key = f"{os.environ['MODAL_API_KEY']}:{os.environ['MODAL_TOKEN_SECRET']}"

client = OpenAI(
    base_url="https://api.modal.com/v1",
    api_key=modal_key
)

try:
    response = client.chat.completions.create(
        model="zai-org/GLM-5.3-Flash",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print("Response type:", type(response))
    print("Response representation:", repr(response))
    if hasattr(response, 'choices'):
        print("Choices:", response.choices)
except Exception as e:
    print(f"Error: {e}")
