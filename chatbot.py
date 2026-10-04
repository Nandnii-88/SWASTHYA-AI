import os
import time
from dotenv import load_dotenv
from groq import Groq, APIError, RateLimitError

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()

# Check GROQ_API_KEY
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GroqAPIKey")

if not API_KEY:
    raise ValueError("GROQ_API_KEY is missing from environment variables")

# ---------------------------------------------------------
# Groq client & Model Configuration
# ---------------------------------------------------------

client = Groq(api_key=API_KEY)

# Better model for detailed Gemini-like answers
PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODELS = ["openai/gpt-oss-20b"]

SYSTEM_INSTRUCTIONS = """
You are SwasthyaAI, a helpful and knowledgeable multilingual health assistant.

LANGUAGE:
- Always reply in the language the user requested (Hindi, English, Bengali, Tamil, etc.) in its native script.

RESPONSE STYLE:
- Give DETAILED, WELL-STRUCTURED answers like ChatGPT/Gemini.
- Use headings, bullet points, and numbered lists when helpful.
- Explain medical concepts clearly with examples.
- If the user describes symptoms, provide:
  1. Possible common causes (not a diagnosis)
  2. General self-care tips
  3. When to see a doctor
  4. Warning signs to watch for
- Do NOT just ask questions back — give helpful information FIRST, then ask 1-2 follow-up questions if needed.

MEDICAL SAFETY:
- Do not give a definitive diagnosis or prescribe specific prescription drugs.
- You MAY suggest general OTC guidance (e.g., "paracetamol for fever, follow package directions").
- For emergencies (chest pain, difficulty breathing, severe bleeding), tell them to call 108/112 immediately.

TONE:
- Warm, clear, and confident. Not robotic or overly cautious.
- Answer like a knowledgeable friend who happens to be a doctor.

GOAL:
Give answers that are USEFUL and COMPLETE, not short and vague. Users should feel they learned something.
"""

# ---------------------------------------------------------
# Main chatbot function
# ---------------------------------------------------------

def ask_groq(message, history="", language="English"):
    """
    Send a message to SwasthyaAI using Groq with fast response time,
    retries, and fallback model support.
    """
    language = str(language).strip() or "English"
    conversation = history.strip() or "(No previous conversation.)"

    user_content = f"""Language to respond in: {language}
(You MUST reply strictly in {language} and using its native script if applicable)

Conversation History:
{conversation}

User: {message}"""

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS.strip()},
        {"role": "user", "content": user_content}
    ]

    models_to_try = [PRIMARY_MODEL] + FALLBACK_MODELS

    for model_name in models_to_try:
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=1200,
                    temperature=0.6
                )

                if response and response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()

            except RateLimitError:
                print(f"[Warning] Rate limited on Groq model {model_name} (attempt {attempt + 1}).")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                break

            except APIError as e:
                print(f"[Warning] Groq API error on model {model_name}: {e}")
                break

            except Exception as e:
                print(f"[Error] Unexpected error on {model_name}: {e}")
                break

    return (
        "I'm sorry, our service is experiencing high demand right now. "
        "Please wait a few moments and try again."
    )
