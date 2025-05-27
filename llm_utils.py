from google import genai
from google.genai import types
from openai import OpenAI
import anthropic
import httpx 
import os 


openai_api_key = os.getenv("OPENAI_API_KEY")
# gemini_api_key = os.getenv("GEMINI_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)
http_client = httpx.Client(
    timeout=httpx.Timeout(120, read=360, connect=20.0),
    transport=httpx.HTTPTransport(retries=2),
    http2=True,
)
gemini_client = genai.Client(api_key= "", http_options= http_client)

def get_openai_response(messages, model="o3"):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


def get_gemini_response(messages, model="gemini-2.5-pro-preview-05-06"):
    cache = gemini_client.caches.create(
        model=model,
        config=types.CreateCachedContentConfig(
            display_name='CACHE_1',
            contents=[messages],
            ttl="300s",
        )
    )

    try:
        response = gemini_client.models.generate_content(
            model = model,
            contents= ("Generate the guide"),
            config=types.GenerateContentConfig(cached_content=cache.name)
        )
    except:
        gemini_client.caches.delete(cache.name)

    # response = gemini_client.models.generate_content(
    #     model=model,
    #     contents=messages,
    # )

    return response.text


def get_claude_completion(
    prompt: str,
    model: str = "claude-opus-4-20250514",
) -> str:
    client = anthropic.Anthropic(
        api_key=""
    )

    message = client.messages.create(
        model=model,
        max_tokens=8000,
        temperature=1,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    return "".join(block.text for block in message.content if block.type == "text")
