import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found in .env")

# Connect to Gemini
client = genai.Client(api_key=api_key)

question = "Explain Artificial Intelligence using a creative example."

# Compare different temperatures
temperatures = [0.2, 1.0, 1.5]

for temp in temperatures:
    print("\n" + "=" * 40)
    print("TEMPERATURE:", temp)
    print("=" * 40)

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=question,
        config=types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=200
        )
    )

    print(response.text)