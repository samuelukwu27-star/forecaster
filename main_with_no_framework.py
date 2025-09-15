# ./main_with_no_framework.py
# A simpler entry point, designed to run the Top Model Bot without
# the overhead of a larger framework.

import os
import sys
from dotenv import load_dotenv

# --- FIX for ModuleNotFoundError ---
# Add the project root directory to the Python path.
# This ensures that the 'bots' module can be found regardless of how the script is executed.
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
# ------------------------------------

from forecaster.bots.top_model_bot import TopModelBot


# Load environment variables from .env file for local development
load_dotenv()

def get_new_questions_from_metaculus():
    """
    Placeholder function to simulate fetching questions.
    In production, this should make an API call to Metaculus.
    """
    print("Fetching new questions from Metaculus...")
    return [
        {
            "id": "minibench",
            "title": "Will global EV sales surpass 30% of all car sales in 2030?",
            "type": "binary",
            "background": "Government incentives and battery tech advancements are key drivers.",
            "resolution_criteria": "Based on IEA's annual report for 2030."
        }
    ]

def submit_forecast_to_metaculus(question_id, probability, rationale):
    """
    Placeholder function for submitting a forecast.
    In production, this should make an API call to Metaculus.
    """
    print("\n--- (SIMULATED) SUBMITTING FORECAST ---")
    print(f"Question ID: {question_id}")
    print(f"Probability: {probability*100:.1f}%")
    print(f"Rationale Snippet: {rationale[:200].strip()}...")
    print("--------------------------------------\n")
    return True

if __name__ == "__main__":
    print("--- Starting Top Model Bot (No Framework) ---")
    
    if not os.getenv("METACULUS_TOKEN"):
        raise ValueError("METACULUS_TOKEN is not set. Check your repository secrets.")

    bot = TopModelBot()
    questions = get_new_questions_from_metaculus()

    for question in questions:
        if question['type'] == 'binary':
            probability, rationale = bot.forecast_question(question)
            
            if probability is not None:
                submit_forecast_to_metaculus(question['id'], probability, rationale)
            else:
                print(f"Failed to produce a forecast for question {question['id']}.")
    
    print("--- Top Model Bot (No Framework) finished. ---")

