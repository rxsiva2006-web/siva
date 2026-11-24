import json
from google import genai
from pydantic import BaseModel

# config.json கோப்பிலிருந்து API Key-ஐப் படித்தல்
with open("config.json", "r") as f:
    config = json.load(f)

client = genai.Client(api_key=config["GEMINI_API_KEY"])

class Summary(BaseModel):
    topic: str
    key_point: str

response = client.interactions.create(
    model="gemini-3.8-flash",
    input="API key பற்றி சுருக்கமாக விளக்கு.",
    response_format={
        "type": "text",
        "mime_type": "application/json",
        "schema": Summary.model_json_schema(),
    },
)

print(response.output_text)