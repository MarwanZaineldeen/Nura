import os
import re
import json
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"
INTENTS      = ["greeting", "goodbye", "gratitude", "asking_mental_health_question", "out_of_scope"]

FEW_SHOT_EXAMPLES = """
Examples:
User: "Hi there!" → greeting
User: "Hello, how are you?" → greeting
User: "Bye, take care!" → goodbye
User: "See you later!" → goodbye
User: "Thanks so much for your help!" → gratitude
User: "I really appreciate it, thank you." → gratitude
User: "I've been feeling really anxious lately and can't sleep." → asking_mental_health_question
User: "How do I cope with depression?" → asking_mental_health_question
User: "I feel so overwhelmed and stressed all the time." → asking_mental_health_question
User: "What are the symptoms of PTSD?" → asking_mental_health_question
User: "What's the capital of France?" → out_of_scope
User: "Can you write me a poem?" → out_of_scope
User: "What's 2 + 2?" → out_of_scope
""".strip()

SYSTEM_PROMPT = f"""You are an intent classification engine for a mental health support chatbot.

Classify the user's message into exactly one of these intents:
- greeting: The user is saying hello or starting a conversation.
- goodbye: The user is ending the conversation.
- gratitude: The user is expressing thanks.
- asking_mental_health_question: The user is asking about mental health, emotions, stress, anxiety, depression, therapy, coping, or related topics.
- out_of_scope: The message does not fit any of the above categories.

{FEW_SHOT_EXAMPLES}

Respond with ONLY a JSON object in this exact format:
{{"intent": "<intent>", "confidence": <0.0-1.0>, "reason": "<one short sentence>"}}

No extra text. No markdown. No explanation outside the JSON."""

client = Groq(api_key=GROQ_API_KEY)


def classify_intent(user_message: str) -> dict:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=150,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )
    raw = response.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            result = {"intent": "out_of_scope", "confidence": 0.0, "reason": "Failed to parse response."}
    if result.get("intent") not in INTENTS:
        result["intent"] = "out_of_scope"
    return result


if __name__ == "__main__":
    test_messages = [
        "Hey there!",
        "Goodbye, thanks for everything.",
        "Thank you so much, this really helped.",
        "I've been having panic attacks and don't know what to do.",
        "How do I cope with anxiety at work?",
        "What's the best programming language to learn?",
        "I feel completely hopeless lately.",
    ]
    for msg in test_messages:
        print(f"\nUser: {msg}")
        result = classify_intent(msg)
        print(f"  Intent: {result['intent']} ({result['confidence']:.0%}) — {result['reason']}")