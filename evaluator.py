import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Check your .env file."
    )

client = Groq(api_key=API_KEY)


def _ask_json(system_prompt, user_prompt, temperature=0.2):
    """Call Groq and parse its JSON response."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("The AI returned an empty response.")

    return json.loads(content)


def generate_questions(
    role,
    experience,
    difficulty,
    question_count=5,
    question_type="Mixed",
):
    """Generate interview questions tailored to the candidate."""

    system_prompt = """
    You are an experienced professional interviewer.
    Generate realistic, relevant interview questions.
    Return only valid JSON. Do not include answers.
    """

    user_prompt = f"""
    Generate exactly {question_count} interview questions.

    Role: {role}
    Experience: {experience}
    Difficulty: {difficulty}
    Type: {question_type}

    Return this structure:
    {{
      "questions": [
        {{
          "question": "Question text",
          "category": "Technical",
          "difficulty": "{difficulty}"
        }}
      ]
    }}

    Requirements:
    - Generate the requested number of distinct questions.
    - Tailor questions to the role and experience level.
    - For Mixed, combine technical, behavioral, and situational
      questions where appropriate.
    - For other types, focus on the selected category.
    - Do not provide answers or hints.
    """

    data = _ask_json(
        system_prompt,
        user_prompt,
        temperature=0.4,
    )

    questions = data.get("questions", [])

    cleaned = []

    for item in questions:
        if isinstance(item, str):
            cleaned.append({
                "question": item,
                "category": question_type,
                "difficulty": difficulty,
            })

        elif isinstance(item, dict) and item.get("question"):
            cleaned.append({
                "question": str(item["question"]),
                "category": str(
                    item.get("category", question_type)
                ),
                "difficulty": str(
                    item.get("difficulty", difficulty)
                ),
            })

    if len(cleaned) < question_count:
        raise ValueError(
            f"The AI generated only {len(cleaned)} valid questions. "
            "Please try generating them again."
        )

    return cleaned[:question_count]


def evaluate_answer(question, answer, role, experience):
    """Evaluate a candidate's answer and return structured feedback."""

    system_prompt = """
    You are a fair, constructive interview coach.

    Evaluate the answer for relevance, correctness, clarity,
    structure, and supporting examples. Adjust expectations
    to the candidate's experience level.

    Return only valid JSON with these fields:
    {
      "score": 75,
      "strengths": ["A specific strength"],
      "improvements": ["A specific improvement"],
      "ideal_answer": "An example of a stronger answer",
      "summary": "A concise overall assessment"
    }

    Score must be an integer between 0 and 100.
    Be specific and respectful. Never invent candidate experience.
    """

    user_prompt = f"""
    Job role: {role}
    Experience level: {experience}

    Interview question:
    {question}

    Candidate's answer:
    {answer.strip() if answer.strip() else "[No answer provided]"}

    Evaluate this answer fairly and provide actionable feedback.
    """

    data = _ask_json(
        system_prompt,
        user_prompt,
        temperature=0.1,
    )

    try:
        score = int(data.get("score", 0))
    except (ValueError, TypeError):
        score = 0

    strengths = data.get("strengths", [])
    improvements = data.get("improvements", [])

    if not isinstance(strengths, list):
        strengths = []

    if not isinstance(improvements, list):
        improvements = []

    return {
        "score": max(0, min(100, score)),
        "strengths": strengths,
        "improvements": improvements,
        "ideal_answer": str(data.get("ideal_answer", "")),
        "summary": str(data.get("summary", "")),
    }