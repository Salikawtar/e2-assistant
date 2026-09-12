import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

key = os.getenv("GEMINI_API_KEY")
if not key:
    raise SystemExit("No GEMINI_API_KEY found — check your .env file.")

print(f"Key loaded: {key[:6]}...{key[-4:]}")

client = genai.Client(api_key=key)

result = client.models.embed_content(
    model="gemini-embedding-2",
    contents="anhydrous ammonia minimum quantity threshold",
    config=types.EmbedContentConfig(output_dimensionality=768),
)

vector = result.embeddings[0].values
print(f"Vector length: {len(vector)}")
print(f"First 5 numbers: {vector[:5]}")