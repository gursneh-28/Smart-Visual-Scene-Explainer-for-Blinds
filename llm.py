from google import genai
from google.genai import types
import os
import base64
from dotenv import load_dotenv

load_dotenv()

# Keep track of the last few things said to avoid repetition
# In a real app, this might be per-session
last_descriptions = []

def generate_description(image_bytes, spatial_descriptions, texts, find_object=None, conversation_history=None):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("LLM error: No API key found")
        return None

    client = genai.Client(api_key=api_key)
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    # Build context from detected objects
    object_context = ""
    if spatial_descriptions:
        lines = [f"- {obj['display_name']}: {obj['horizontal']}, {obj['distance_str']}" for obj in spatial_descriptions[:8]]
        object_context = "\nScene Data:\n" + "\n".join(lines)

    text_context = ""
    if texts:
        text_context = "\nVisible Text: " + ", ".join([f'"{t["text"]}"' for t in texts[:5]])

    # System instruction for a premium, assistive AI persona
    system_instruction = """
    You are 'Vision' — a high-end AI assistant for the visually impaired.
    Your goal is to be a helpful, calm, and sophisticated companion, NOT a robotic scanner.
    
    GUIDELINES:
    1. BE CONVERSATIONAL: Speak like a human. Use "I see...", "There is...", "To your left, you'll find...".
    2. BE CONCISE: Never use more than 2-3 sentences unless asked for detail.
    3. DON'T BE ANNOYING: If the scene is the same as before, be very brief or mention only changes.
    4. SAFETY FIRST: Always mention obstacles or hazards (stairs, people, cars) immediately.
    5. INTERACTIVE: End with a helpful suggestion or an invitation for a question if appropriate.
    6. DISTANCE: Always use metric distances (metres).
    """

    if find_object and find_object != 'text_only':
        prompt = f"The user is looking for: '{find_object}'.\n{object_context}\n{text_context}\nTell them if you see it and exactly where, or guide them to find it."
    elif find_object == 'text_only':
        prompt = f"Read the text in this image clearly and naturally.\n{text_context}"
    else:
        prompt = f"Describe the current scene naturally.\n{object_context}\n{text_context}\nFocus on what's most relevant to someone walking or sitting here."

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4, # Lower temperature for more stable, less "random" descriptions
            ),
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_b64)),
                        types.Part(text=prompt)
                    ]
                )
            ]
        )
        return response.text.strip()
    except Exception as e:
        print(f"LLM error: {e}")
        return None