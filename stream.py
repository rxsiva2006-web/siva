import os
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Ask Gemini a question
question = "Explain Generative AI in simple words."

print("Gemini is responding:\n")

# Generate response using streaming
response = client.models.generate_content_stream(
    model="gemini-3.5-flash",
    contents=question
)

# Display each chunk immediately
for chunk in response:
    if chunk.text:
        print(chunk.text, end="", flush=True)

print("\n\nResponse completed!")