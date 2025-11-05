# request to timeweb llm api
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv("api_key.env")

url = os.getenv("timeweb_openai_url")
api_key = os.getenv("timeweb_api")

class TimewebLLMRequest:
    def __init__(self, message: str, parent_message_id: str):
        self.message = message
        self.parent_message_id = parent_message_id
        headers = {
        "Authorization": "",
        "Content-Type": "application/json"
        }

        json = {
            "message": self.message,
            "parent_message_id": self.parent_message_id
        }

    def send_request(self):
        response = requests.post(url, headers=self.headers, json=self.json)
        return response.json()
