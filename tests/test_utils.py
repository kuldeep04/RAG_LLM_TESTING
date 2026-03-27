import json
from pathlib import Path

import requests


def load_test_data(filename):
    test_data_path = Path(__file__).parent / "testdata"/filename
    with open(test_data_path) as json_file:
        return json.load(json_file)

def get_llm_response(test_data):
    return requests.post("https://rahulshettyacademy.com/rag-llm/ask",
                  json={
                      "question": test_data["question"],
                      "chat_history": []
                  }).json()
