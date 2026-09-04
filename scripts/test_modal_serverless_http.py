import os
import requests
from dotenv import load_dotenv

load_dotenv()
modal_key = f"{os.environ['MODAL_API_KEY']}:{os.environ['MODAL_TOKEN_SECRET']}"
headers = {
    "Authorization": f"Bearer {modal_key}", # Or maybe Basic auth?
    "Content-Type": "application/json"
}

print("Testing with Bearer token...")
response = requests.post(
    "https://api.modal.com/v1/chat/completions",
    headers=headers,
    json={
        "model": "zai-org/GLM-5.3-Flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.0,
        "seed": 42
    }
)
print("Status:", response.status_code)
print("Text:", response.text)

print("\nTesting with Basic auth...")
import base64
auth_string = base64.b64encode(modal_key.encode()).decode()
headers["Authorization"] = f"Basic {auth_string}"
response = requests.post(
    "https://api.modal.com/v1/chat/completions",
    headers=headers,
    json={
        "model": "zai-org/GLM-5.3-Flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.0,
        "seed": 42
    }
)
print("Status:", response.status_code)
print("Text:", response.text)
