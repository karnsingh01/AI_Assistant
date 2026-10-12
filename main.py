
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
import os
import time
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# Use a model available to your Gemini API key
MODEL_NAME = "gemini-3.8-flash"


def get_ai_response(system_prompt, user_prompt):
    """Generate an AI response with retries for temporary errors."""

    prompt = (
        f"{system_prompt}\n\n"
        f"User request:\n{user_prompt}"
    )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return response.text.strip()

        except Exception as e:
            error_message = str(e).upper()

            is_temporary_error = (
                "503" in error_message
                or "UNAVAILABLE" in error_message
                or "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
                or "500" in error_message
                or "INTERNAL" in error_message
            )

            if is_temporary_error and attempt < 2:
                wait_seconds = 2 ** (attempt + 1)
                app.logger.warning(
                    "Temporary Gemini error. Retrying in %s seconds "
                    "(attempt %s/3).",
                    wait_seconds,
                    attempt + 1
                )
                time.sleep(wait_seconds)
                continue

            raise


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()

    if not question:
        return jsonify({
            "error": "Please enter a question."
        }), 400

    try:
        answer = get_ai_response(
            "You are a helpful AI personal assistant. "
            "Give clear, accurate and useful answers.",
            question
        )

        return jsonify({
            "response": answer
        }), 200

    except Exception:
        app.logger.exception("Error in /ask")

        return jsonify({
            "error": (
                "Unable to generate an answer right now. "
                "Please try again later."
            )
        }), 503


@app.route("/summarize", methods=["POST"])
def summarize():
    email_text = request.form.get("email", "").strip()

    if not email_text:
        return jsonify({
            "error": "Please paste an email to summarize."
        }), 400

    try:
        summary = get_ai_response(
            "You are an expert email assistant. "
            "Summarize the provided email in 2-3 clear sentences. "
            "Preserve important dates, actions and key details.",
            email_text
        )

        return jsonify({
            "response": summary
        }), 200

    except Exception:
        app.logger.exception("Error in /summarize")

        return jsonify({
            "error": (
                "Unable to summarize the email right now. "
                "Please try again later."
            )
        }), 503


if __name__ == "__main__":
    app.run(debug=True)
