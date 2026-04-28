from google import genai
from google.genai import types
import os
import time
from dotenv import load_dotenv

load_dotenv()

# ── Singleton client — don't recreate per request ──
_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        _client = genai.Client(api_key=api_key)
    return _client

# ── System prompt — compact but rich context ──
SYSTEM_PROMPT = """You are Vision — an AI assistant for the visually impaired.
RULES (follow strictly):
1. Respond in 1-3 short sentences MAX. Be concise. Do not ramble.
2. Speak naturally like a helpful human: "I see...", "To your left...", "There's a..."
3. ALWAYS mention safety hazards first (stairs, cars, people rushing).
4. Use metric distances. Avoid technical jargon.
5. For "find" requests: say exactly where the object is, or say you cannot see it.
6. For text: read it naturally in order, skip unimportant symbols.
7. Never say "I am an AI" or "as Vision" — just respond directly."""

def generate_description(
    image_bytes: bytes,
    spatial_descriptions: list,
    texts: list,
    find_object: str | None = None,
    max_retries: int = 2
) -> str | None:
    client = _get_client()

    # ── Build a lean context string ──
    obj_ctx = ""
    if spatial_descriptions:
        lines = [
            f"{obj['display_name']} ({obj['horizontal']}, ~{obj.get('distance_m', '?')}m)"
            for obj in spatial_descriptions[:6]
        ]
        obj_ctx = "Detected: " + "; ".join(lines)

    txt_ctx = ""
    if texts:
        txt_ctx = "Text visible: " + ", ".join(f'"{t["text"]}"' for t in texts[:5])

    # ── Build targeted prompt ──
    if find_object and find_object != 'text_only':
        prompt = (
            f"User is looking for: {find_object!r}.\n"
            f"{obj_ctx}\n"
            "Look at the image carefully. Tell them precisely where the object is "
            "(left/right/centre, near/far). If not visible, say so clearly."
        )
    elif find_object == 'text_only':
        prompt = (
            f"{txt_ctx}\n"
            "Read ALL visible text in the image, in natural reading order. "
            "Skip decorative symbols. Speak naturally."
        )
    else:
        prompt = (
            f"{obj_ctx}\n"
            f"{txt_ctx}\n"
            "Look at the image. Describe the most important things for someone who cannot see. "
            "Mention safety concerns first, then the main objects and their locations."
        )

    # ── Call Gemini with retry logic ──
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=150,  # Force brevity — no walls of text
                ),
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            types.Part.from_text(text=prompt),
                        ],
                    )
                ],
            )
            text = response.text.strip()
            return text if text else None

        except Exception as e:
            err = str(e)
            # Rate limit → wait and retry
            if "429" in err or "quota" in err.lower():
                wait = 2 ** attempt  # Exponential backoff: 1s, 2s
                time.sleep(wait)
                continue
            # Other errors → fail fast
            print(f"LLM error (attempt {attempt + 1}): {e}")
            if attempt == max_retries:
                return None

    return None

def generate_description_stream(image_bytes, spatial_descriptions, texts, find_object=None):
    """
    Generator that yields text chunks as they arrive from the API.
    For future use with SSE streaming endpoint.
    """
    client = _get_client()

    obj_ctx = ""
    if spatial_descriptions:
        lines = [f"{obj['display_name']} ({obj['horizontal']})" for obj in spatial_descriptions[:5]]
        obj_ctx = "Detected: " + "; ".join(lines)

    txt_ctx = ""
    if texts:
        txt_ctx = "Text: " + ", ".join(f'"{t["text"]}"' for t in texts[:4])

    prompt = f"{obj_ctx}\n{txt_ctx}\nDescribe the scene concisely for a visually impaired person."

    try:
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=150,
            ),
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=prompt),
                    ],
                )
            ],
        ):
            if chunk.text:
                yield chunk.text
    except Exception as e:
        print(f"Stream error: {e}")
        yield None