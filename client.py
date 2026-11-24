import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_ID = "gemini-3.5-flash"
CONFIG = types.GenerateContentConfig(
    system_instruction="You are a helpful assistant.",
)

def create_chat(history=None):
    return client.chats.create(
        model=MODEL_ID,
        config=CONFIG,
        history=history or [],
    )

chat = create_chat()
turn_count = 0
total_session_tokens = 0
MAX_TURNS = 3

print("Chat started! Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in {"exit", "quit"}:
        print(f"\n[Session Ended | Total Tokens Used: {total_session_tokens}]")
        break

    response = chat.send_message(user_input)
    print(f"\nGemini: {response.text}")

    # 1. Extract token usage for this turn
    usage = response.usage_metadata
    turn_tokens = usage.total_token_count if usage else 0
    total_session_tokens += turn_tokens

    # 2. Check current active history token size
    history_token_info = client.models.count_tokens(
        model=MODEL_ID,
        contents=chat.get_history()
    )

    print(
        f"[Tokens -> Input: {usage.prompt_token_count} | "
        f"Output: {usage.candidates_token_count} | "
        f"Turn Total: {turn_tokens} | "
        f"Active Memory: {history_token_info.total_tokens}]\n"
    )

    turn_count += 1

    # Clear older history every 3rd turn while keeping the last 2 turns
    if turn_count % MAX_TURNS == 0:
        recent_history = chat.get_history()[-4:]
        chat = create_chat(history=recent_history)

        trimmed_info = client.models.count_tokens(
            model=MODEL_ID,
            contents=chat.get_history()
        )
        print(
            f"[Info: Old chat tokens cleared! "
            f"Active Memory reduced to: {trimmed_info.total_tokens} tokens]\n"
        )