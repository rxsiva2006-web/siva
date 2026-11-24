import os
from dotenv import load_dotenv
from google import genai
from google.genai import errors

# 1. Load environment variables from your .env file
load_dotenv()

# 2. Check that the API key exists before making a request
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found! Make sure it is set in your .env file.")

# 3. Initialize the Gemini client
client = genai.Client(api_key=api_key)

def ask_gemini(prompt: str, model: str = "gemini-3.5-flash") -> str:
    """Sends a prompt to Gemini and returns the text response."""
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text
    except errors.APIError as e:
        return f"API Error: {e.message} (Status Code: {e.code})"
    except Exception as e:
        return f"Unexpected Error: {e}"

if __name__ == "__main__":
    test_prompt = "Hello! Explain in one short sentence what you can do."
    print(f"Sending prompt: '{test_prompt}'\n")
    
    reply = ask_gemini(test_prompt)
    print("Gemini Response:")
    print(reply)
