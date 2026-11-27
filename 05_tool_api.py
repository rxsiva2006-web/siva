import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.5-flash")

if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env file.")

client = genai.Client(api_key=API_KEY)
app = FastAPI(title="Week 1 Day 5 — Tool-Calling Support API")

MOCK_ORDERS = {
    "ORD-101": {"status": "Delivered", "item": "Mechanical Keyboard", "delivered_on": "2026-11-24"},
    "ORD-102": {"status": "Shipped", "item": "27-inch 4K Monitor", "eta": "2026-11-29", "carrier": "BlueDart"},
    "ORD-103": {"status": "Processing", "item": "USB-C Docking Station", "eta": "2026-12-01"},
    "ORD-104": {"status": "Cancelled", "item": "Wireless Mouse", "refund_status": "Refunded to card"},
}


# TODO 1: Add a clear docstring explaining what this tool does and what argument it expects
def get_order_status(order_id: str) -> dict:
    clean_id = order_id.strip().upper()
    if clean_id in MOCK_ORDERS:
        return {"found": True, "order_id": clean_id, **MOCK_ORDERS[clean_id]}
    return {"found": False, "order_id": clean_id, "message": "Order ID not found in database."}


TOOL_REGISTRY = {
    "get_order_status": get_order_status,
}

SYSTEM_PROMPT = """
You are a helpful e-commerce customer support assistant.
- Use the `get_order_status` tool whenever a user asks about a specific order ID.
- Never guess or invent order statuses. If the tool returns found=False, politely tell the customer the order ID was not found.
- For general questions that do not require an order lookup, answer concisely without calling any tool.
"""


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="Customer question")


class AskResponse(BaseModel):
    answer: str
    tool_called: str | None = None
    tool_args: dict | None = None
    tool_result: dict | None = None
    latency_ms: float


# TODO 2: Add @retry decorator (3 attempts, exponential backoff 1s to 8s, reraise=True)
def generate_with_tools(contents: list, config: types.GenerateContentConfig):
    return client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=config,
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "model": MODEL_NAME}


# TODO 3 & TODO 4: Implement the POST /ask endpoint with manual tool execution and AskResponse output
@app.post("/ask", response_model=AskResponse)
def ask_assistant(payload: AskRequest):
    start = time.perf_counter()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.0,
        tools=[get_order_status],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # 1. Build initial contents list with payload.question
    # 2. Call generate_with_tools(contents, config)
    # 3. If first_response.function_calls exists:
    #      - Extract fc = first_response.function_calls[0]
    #      - Execute tool_output = TOOL_REGISTRY[fc.name](**dict(fc.args))
    #      - Append first_response.candidates[0].content AND Part.from_function_response(...) to contents
    #      - Call generate_with_tools(contents, config) a second time for final_answer
    # 4. Otherwise return first_response.text directly with tool_called=None
    raise NotImplementedError("Implement POST /ask tool-calling endpoint")
