# A forecasting bot using top OpenRouter models for reasoning,
# with research powered by NewsAPI and direct web scraping.
# This script combines robust, paginated data fetching with multi-tournament processing.

import asyncio
import datetime
import json
import logging
import os
import random
import re
import traceback
import uuid

import dotenv
import numpy as np
import requests
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
from openai import AsyncOpenAI  # Used to connect to OpenRouter's compatible API

dotenv.load_dotenv()

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ForecastBot")

######################### CONSTANTS #########################
# --- Core Settings ---
SUBMIT_PREDICTION = True  # Set to True to publish your predictions to Metaculus
NUM_RUNS_PER_QUESTION = 3  # The median forecast is taken between N runs
SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = True
RUN_ID = str(uuid.uuid4())  # Generate a unique ID for this script run

# --- Environment Variables & API Keys ---
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")

# --- API Clients & URLs ---
newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY) if NEWSAPI_API_KEY else None
AUTH_HEADERS = {"Authorization": f"Token {METACULUS_TOKEN}"}
API_V1_BASE_URL = "https://www.metaculus.com/api"  # For submitting predictions/comments
API_V2_BASE_URL = "https://www.metaculus.com/api2"  # For fetching questions

# --- OpenRouter Configuration ---
OPENROUTER_MODELS = ["openai/gpt-4", "openai/gpt-4o"]
APP_URL = "https://github.com/your-repo/forecaster"  # For OpenRouter headers
APP_TITLE = "Metaculus Forecasting Bot"  # For OpenRouter headers

# --- Tournament & Question IDs ---
# A list of tournament slugs to process
TOURNAMENT_IDS = ["fall-aib-2025", "metaculus-cup-fall-2025", "minibench"]

######################### METACULUS API FUNCTIONS #########################

def list_questions_from_tournament(tournament_id: str, count: int = 50) -> list[dict]:
    """
    Fetch all questions from a given Metaculus tournament using pagination (API v2).
    """
    questions = []
    # Note: APIv2 uses 'tournaments__slug' instead of 'tournaments'
    url = f"{API_V2_BASE_URL}/questions/?limit={count}&tournaments__slug={tournament_id}&include_description=true&status=open"
    url += "&forecast_type=binary,multiple_choice,numeric,discrete"

    page = 1
    while url:
        logger.info(f"Fetching page {page} for tournament '{tournament_id}'")
        try:
            resp = requests.get(url, headers=AUTH_HEADERS)
            resp.raise_for_status()

            data = resp.json()
            results = data.get("results", [])
            logger.info(f"Retrieved {len(results)} questions from page {page}")

            questions.extend(results)
            url = data.get("next")  # next page URL or None
            page += 1
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch page {page} for tournament {tournament_id}: {e}")
            break

    logger.info(f"Total retrieved for tournament '{tournament_id}': {len(questions)} questions")
    return questions


def post_question_comment(post_id: int, comment_text: str) -> None:
    response = requests.post(
        f"{API_V1_BASE_URL}/comments/create/",
        json={
            "text": comment_text, "parent": None, "included_forecast": True,
            "is_private": True, "on_post": post_id,
        },
        headers=AUTH_HEADERS,
    )
    if not response.ok:
        raise RuntimeError(f"Failed to post comment: {response.text}")


def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    url = f"{API_V1_BASE_URL}/questions/forecast/"
    response = requests.post(
        url, json=[{"question": question_id, **forecast_payload}], headers=AUTH_HEADERS,
    )
    logger.info(f"Prediction Post status code: {response.status_code}")
    if not response.ok:
        raise RuntimeError(f"Failed to post prediction: {response.text}")

def create_forecast_payload(forecast, question_type: str) -> dict:
    if question_type == "binary":
        return {"probability_yes": forecast}
    if question_type == "multiple_choice":
        return {"probability_yes_per_category": forecast}
    return {"continuous_cdf": forecast}


def forecast_is_already_made(question_details: dict) -> bool:
    try:
        # APIv2 format for personal forecast is slightly different
        return question_details["my_forecasts"]["forecast"] is not None
    except (KeyError, TypeError):
        return False

######################### RESEARCH & LLM FUNCTIONS #########################

CONCURRENT_REQUESTS_LIMIT = 5
llm_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)

async def call_llm(prompt: str, model: str, temperature: float = 0.3) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set.")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    async with llm_rate_limiter:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, stream=False,
                extra_headers={"HTTP-Referer": APP_URL, "X-Title": APP_TITLE}
            )
            answer = response.choices[0].message.content
            if answer is None:
                raise ValueError("No answer returned from LLM")
            return answer
        except Exception as e:
            logger.error(f"[OpenRouter] Error calling LLM ({model}): {e}")
            raise


def perform_web_scrape(query: str) -> str:
    logger.info(f"[Web Scraper] Searching for: {query}")
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        top_result = soup.find('a', class_='result__a')
        if not top_result or 'href' not in top_result.attrs:
            return "Web scraping found no clear top result."
        page_url = top_result['href']
        logger.info(f"[Web Scraper] Scraping content from: {page_url}")
        page_response = requests.get(page_url, headers=headers, timeout=10)
        page_response.raise_for_status()
        page_soup = BeautifulSoup(page_response.text, 'html.parser')
        paragraphs = page_soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs[:5]])
        return f"Web Scrape Summary:\n- {content[:1500]}..."
    except Exception as e:
        return f"Web scraping failed: {e}"


def call_newsapi(question: str) -> str:
    if not newsapi_client:
        logger.warning("[NewsAPI] NewsAPI key not set. Skipping news search.")
        return "NewsAPI search not performed."
    try:
        all_articles = newsapi_client.get_everything(q=question, language='en', sort_by='relevancy', page_size=5)
        if not all_articles or not all_articles['articles']:
            return "No recent news articles found."
        formatted_articles = ""
        for article in all_articles['articles']:
            formatted_articles += f"- Title: {article['title']}\n  Source: {article['source']['name']}\n  Snippet: {article.get('description', 'N/A')}\n"
        return f"Recent News:\n{formatted_articles}"
    except Exception as e:
        return f"NewsAPI search failed: {e}"


def run_research(question: str) -> str:
    logger.info(f"--- Running Research for: {question} ---")
    scrape_results = perform_web_scrape(question)
    news_results = call_newsapi(question)
    research = f"{scrape_results}\n\n{news_results}"
    logger.info(f"--- Research Complete ---\n{research[:300]}...\n--------------------")
    return research

######################### FORECASTING LOGIC #########################

BINARY_PROMPT_TEMPLATE = """
You are a professional forecaster. Your task is to predict the outcome of a binary question.
Question: {title}
Background: {background}
Resolution Criteria: {resolution_criteria}
Today is {today}.
Your research assistant has provided the following summary:
{summary_report}
Before answering, please provide your reasoning in detail.
Finally, provide your probability estimate on a new line in the format: "Probability: ZZ%"
"""

def extract_probability_from_response(forecast_text: str) -> float:
    matches = re.findall(r"(\d+)%", forecast_text)
    if matches:
        return min(99, max(1, int(matches[-1])))  # clamp
    raise ValueError(f"Could not extract prediction from response: {forecast_text}")

async def get_binary_llm_prediction(question_details: dict, num_runs: int) -> tuple[float, str]:
    title = question_details["title"]
    summary_report = run_research(title)
    content = BINARY_PROMPT_TEMPLATE.format(
        title=title, today=datetime.datetime.now().strftime("%Y-%m-%d"),
        background=question_details.get("description", ""),
        resolution_criteria=question_details.get("resolution_criteria", ""),
        summary_report=summary_report,
    )

    async def get_rationale_and_prob(prompt: str, model: str) -> tuple[float, str]:
        rationale = await call_llm(prompt, model=model)
        probability = extract_probability_from_response(rationale)
        comment = f"Model: {model}\nExtracted Probability: {probability}%\n\nLLM's Answer:\n{rationale}"
        return probability, comment

    tasks = [get_rationale_and_prob(content, OPENROUTER_MODELS[i % len(OPENROUTER_MODELS)]) for i in range(num_runs)]
    results = await asyncio.gather(*tasks)
    
    probabilities = [r[0] for r in results]
    median_probability = float(np.median(probabilities)) / 100.0
    
    comments = [r[1] for r in results]
    final_comment_sections = [f"## Run {i+1}\n{c}" for i, c in enumerate(comments)]
    final_comment = f"Median Probability: {median_probability*100:.1f}%\n\n" + "\n\n".join(final_comment_sections)
    
    return median_probability, final_comment

# --- Placeholder Forecasting for Other Question Types ---
async def get_numeric_placeholder_prediction(question_details: dict) -> tuple[dict, str]:
    """Generate a placeholder forecast for a numeric question."""
    low = random.uniform(0, 50)
    high = low + random.uniform(10, 100)
    forecast = {"low": low, "high": high} # NOTE: This is NOT the correct final format for submission
    comment = f"Placeholder numeric forecast for Q{question_details['id']}: {low:.1f}–{high:.1f}"
    logger.debug(comment)
    return forecast, comment

async def get_multiple_choice_placeholder_prediction(question_details: dict) -> tuple[dict, str]:
    """Generate a placeholder forecast for a multiple-choice question."""
    options = question_details.get("possibilities", {}).get("options", [])
    weights = [random.random() for _ in options]
    total = sum(weights)
    probs = [w / total for w in weights]
    # The API expects a dictionary mapping option names to probabilities
    forecast = {opt["name"]: p for opt, p in zip(options, probs)}
    comment = f"Placeholder MC forecast for Q{question_details['id']}: {forecast}"
    logger.debug(comment)
    return forecast, comment


################### MAIN FORECASTING LOOP ###################

async def forecast_individual_question(question_details: dict) -> str:
    title = question_details["title"]
    question_id = question_details["id"]
    question_type = question_details.get("possibilities", {}).get("type")
    
    # Safely get the URL for logging, providing a default if it's missing.
    url = question_details.get("url", "No URL found")
    summary = f"-----------------------------------------------\nQuestion: {title}\nURL: {url}\n"

    try:
        if SKIP_PREVIOUSLY_FORECASTED_QUESTIONS and forecast_is_already_made(question_details):
            summary += "Skipped: Forecast already made\n"
            logger.info(summary)
            return summary

        if question_type == "binary":
            forecast, comment = await get_binary_llm_prediction(question_details, NUM_RUNS_PER_QUESTION)
        # elif question_type in ("continuous", "date"): # Metaculus API uses 'continuous' for numeric
        #     forecast, comment = await get_numeric_placeholder_prediction(question_details)
        # elif question_type == "multiple_choice":
        #     forecast, comment = await get_multiple_choice_placeholder_prediction(question_details)
        else:
            summary += f"Skipped: Question type '{question_type}' is not yet supported for forecasting.\n"
            logger.info(summary)
            return summary

        summary += f"Forecast: {forecast}\nComment:\n{comment[:200]}...\n"
        
        if SUBMIT_PREDICTION and forecast is not None:
            payload = create_forecast_payload(forecast, question_type)
            post_question_prediction(question_id, payload)
            # Use the question_id for posting comments, as page_id is not available in APIv2
            post_question_comment(question_id, comment)
            summary += "SUCCESS: Prediction and comment submitted.\n"
        else:
            summary += "SKIPPED SUBMISSION.\n"
    
    except Exception:
        summary += f"\n!!!!!!\nCRITICAL ERROR processing question: {title}\n!!!!!!\n"
        # Print the full stack trace for detailed debugging
        logger.error(f"Traceback for question {question_id}:\n{traceback.format_exc()}")

    logger.info(summary)
    return summary

async def main():
    logger.info("--- Starting Unified Forecasting Bot ---")
    if not METACULUS_TOKEN:
        raise ValueError("METACULUS_TOKEN environment variable not set.")

    all_questions = []
    for tid in TOURNAMENT_IDS:
        all_questions.extend(list_questions_from_tournament(tid))

    if not all_questions:
        logger.warning("No open questions found across all specified tournaments.")
        return
        
    logger.info(f"Found a total of {len(all_questions)} questions to process.")

    tasks = [forecast_individual_question(q) for q in all_questions]
    await asyncio.gather(*tasks)
    
    logger.info("--- Forecasting Bot finished. ---")


if __name__ == "__main__":
    asyncio.run(main())

