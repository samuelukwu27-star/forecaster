# A forecasting bot using top OpenRouter models for reasoning,
# with research powered by NewsAPI and direct web scraping.

import asyncio
import datetime
import json
import os
import re
import uuid

import dotenv
import numpy as np
import requests
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
from openai import AsyncOpenAI # Used to connect to OpenRouter's compatible API

dotenv.load_dotenv()

######################### CONSTANTS #########################
# --- Core Settings ---
SUBMIT_PREDICTION = True  # Set to True to publish your predictions to Metaculus
USE_EXAMPLE_QUESTIONS = False  # Set to True to forecast example questions
NUM_RUNS_PER_QUESTION = 3  # The median forecast is taken between N runs
SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = True
RUN_ID = str(uuid.uuid4()) # Generate a unique ID for this script run

# --- Environment Variables & API Keys ---
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")

# --- API Clients ---
newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY) if NEWSAPI_API_KEY else None

# --- OpenRouter Configuration ---
# Using the best models available on OpenRouter, as requested.
# "o1" is interpreted as openai/gpt-4o, a top-tier omni model.
OPENROUTER_MODELS = ["openai/gpt-4", "openai/gpt-4o"]
APP_URL = "https://github.com/your-repo/forecaster" # For OpenRouter headers
APP_TITLE = "Metaculus Forecasting Bot" # For OpenRouter headers


# --- Tournament & Question IDs ---
FALL_2025_AI_BENCHMARKING_ID = "fall-aib-2025"
TOURNAMENT_ID = FALL_2025_AI_BENCHMARKING_ID

EXAMPLE_QUESTIONS = [
    (578, 578),  # Human Extinction - Binary
    (14333, 14333),  # Age of Oldest Human - Numeric
    (22427, 22427),  # Number of New Leading AI Labs - Multiple Choice
]

######################### HELPER FUNCTIONS #########################

AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api"

def post_question_comment(post_id: int, comment_text: str) -> None:
    response = requests.post(
        f"{API_BASE_URL}/comments/create/",
        json={
            "text": comment_text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": post_id,
        },
        **AUTH_HEADERS,
    )
    if not response.ok:
        raise RuntimeError(f"Failed to post comment: {response.text}")

def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    url = f"{API_BASE_URL}/questions/forecast/"
    response = requests.post(
        url,
        json=[{"question": question_id, **forecast_payload}],
        **AUTH_HEADERS,
    )
    print(f"Prediction Post status code: {response.status_code}")
    if not response.ok:
        raise RuntimeError(f"Failed to post prediction: {response.text}")

def create_forecast_payload(forecast, question_type: str) -> dict:
    if question_type == "binary":
        return {"probability_yes": forecast}
    if question_type == "multiple_choice":
        return {"probability_yes_per_category": forecast}
    # numeric or date
    return {"continuous_cdf": forecast}

def list_posts_from_tournament(tournament_id: str = TOURNAMENT_ID, offset: int = 0, count: int = 50) -> list[dict]:
    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": "binary,multiple_choice,numeric,discrete",
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)
    if not response.ok:
        raise Exception(response.text)
    return response.json()

def get_open_question_ids_from_tournament() -> list[tuple[int, int]]:
    posts_data = list_posts_from_tournament()
    open_questions = []
    for post in posts_data.get("results", []):
        if question := post.get("question"):
            if question.get("status") == "open":
                print(f"ID: {question['id']}\nQ: {question['title']}\nCloses: {question['scheduled_close_time']}")
                open_questions.append((question["id"], post["id"]))
    return open_questions

def get_post_details(post_id: int) -> dict:
    url = f"{API_BASE_URL}/posts/{post_id}/"
    print(f"Getting details for {url}")
    response = requests.get(url, **AUTH_HEADERS)
    if not response.ok:
        raise Exception(response.text)
    return response.json()

CONCURRENT_REQUESTS_LIMIT = 5
llm_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)

async def call_llm(prompt: str, model: str, temperature: float = 0.3) -> str:
    """Makes a request to a specific model via the OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    # The OpenAI library is used to connect to OpenRouter's compatible API
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    async with llm_rate_limiter:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=False,
                extra_headers={
                    "HTTP-Referer": APP_URL,
                    "X-Title": APP_TITLE,
                }
            )
            answer = response.choices[0].message.content
            if answer is None:
                raise ValueError("No answer returned from LLM")
            return answer
        except Exception as e:
            print(f"[OpenRouter] Error calling LLM ({model}): {e}")
            raise

# --- Research Functions ---

def perform_web_scrape(query: str) -> str:
    """Performs a web search using DuckDuckGo and scrapes the top result."""
    print(f"[Web Scraper] Searching for: {query}")
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
        print(f"[Web Scraper] Scraping content from: {page_url}")
        page_response = requests.get(page_url, headers=headers, timeout=10)
        page_response.raise_for_status()
        
        page_soup = BeautifulSoup(page_response.text, 'html.parser')
        paragraphs = page_soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs[:5]]) # Get first 5 paragraphs
        
        return f"Web Scrape Summary:\n- {content[:1500]}..." # Limit content size
    except Exception as e:
        return f"Web scraping failed: {e}"


def call_newsapi(question: str) -> str:
    """Fetches recent news articles from NewsAPI."""
    if not newsapi_client:
        print("[NewsAPI] NewsAPI key not set. Skipping news search.")
        return "NewsAPI search not performed."
    try:
        all_articles = newsapi_client.get_everything(
            q=question, language='en', sort_by='relevancy', page_size=5
        )
        if not all_articles or not all_articles['articles']:
            return "No recent news articles found."

        formatted_articles = ""
        for article in all_articles['articles']:
            formatted_articles += f"- Title: {article['title']}\n  Source: {article['source']['name']}\n  Snippet: {article.get('description', 'N/A')}\n"
        return f"Recent News:\n{formatted_articles}"
    except Exception as e:
        return f"NewsAPI search failed: {e}"

def run_research(question: str) -> str:
    """Orchestrates research calls and combines results."""
    print(f"--- Running Research for: {question} ---")
    scrape_results = perform_web_scrape(question)
    news_results = call_newsapi(question)
    
    research = f"{scrape_results}\n\n{news_results}"
    print(f"--- Research Complete ---\n{research[:300]}...\n--------------------")
    return research

############### BINARY ###############

BINARY_PROMPT_TEMPLATE = """
You are a professional forecaster. Your task is to predict the outcome of a binary question.

Question: {title}
Background: {background}
Resolution Criteria: {resolution_criteria}
Additional Details (Fine Print): {fine_print}

Your research assistant has provided the following summary:
{summary_report}

Today is {today}.

Before answering, please provide your reasoning:
1.  Analyze the key factors and their likely impact.
2.  Consider the status quo and what it would take to change it.
3.  Describe a plausible scenario for a "Yes" outcome and a "No" outcome.
4.  Conclude with your rationale.

Finally, provide your probability estimate on a new line in the format: "Probability: ZZ%"
"""

def extract_probability_from_response_as_percentage_not_decimal(forecast_text: str) -> float:
    matches = re.findall(r"(\d+)%", forecast_text)
    if matches:
        number = int(matches[-1])
        return min(99, max(1, number))  # clamp
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")

async def get_binary_gpt_prediction(question_details: dict, num_runs: int) -> tuple[float, str]:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    
    summary_report = run_research(title)
    
    content = BINARY_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=question_details.get("description", ""),
        resolution_criteria=question_details.get("resolution_criteria", ""),
        fine_print=question_details.get("fine_print", ""),
        summary_report=summary_report,
    )

    async def get_rationale_and_probability(prompt: str, model: str) -> tuple[float, str]:
        rationale = await call_llm(prompt, model=model)
        probability = extract_probability_from_response_as_percentage_not_decimal(rationale)
        comment = f"Model: {model}\nExtracted Probability: {probability}%\n\nLLM's Answer:\n{rationale}"
        return probability, comment

    # Create tasks, cycling through the specified models
    tasks = []
    for i in range(num_runs):
        model_to_use = OPENROUTER_MODELS[i % len(OPENROUTER_MODELS)]
        tasks.append(get_rationale_and_probability(content, model_to_use))
        
    results = await asyncio.gather(*tasks)
    
    probabilities = [r[0] for r in results]
    comments = [r[1] for r in results]
    
    median_probability = float(np.median(probabilities)) / 100
    
    final_comment_sections = [f"## Run {i+1}\n{c}" for i, c in enumerate(comments)]
    final_comment = f"Median Probability: {median_probability*100:.1f}%\n\n" + "\n\n".join(final_comment_sections)
    
    return median_probability, final_comment
    
# NOTE: Placeholder functions for NUMERIC and MULTIPLE_CHOICE are omitted for brevity.
# The core logic would be similar: create a prompt, call the LLM, parse the structured output.

################### FORECASTING ###################

def forecast_is_already_made(post_details: dict) -> bool:
    try:
        return post_details["question"]["my_forecasts"]["latest"]["forecast_values"] is not None
    except (KeyError, TypeError):
        return False

async def forecast_individual_question(
    question_id: int, post_id: int, submit_prediction: bool, 
    num_runs_per_question: int, skip_previously_forecasted_questions: bool
) -> str:
    post_details = get_post_details(post_id)
    question_details = post_details["question"]
    title = question_details["title"]
    question_type = question_details["type"]

    summary = f"-----------------------------------------------\nQuestion: {title}\nURL: https://www.metaculus.com/questions/{post_id}/\n"

    if skip_previously_forecasted_questions and forecast_is_already_made(post_details):
        summary += "Skipped: Forecast already made\n"
        print(summary)
        return summary

    if question_type == "binary":
        forecast, comment = await get_binary_gpt_prediction(question_details, num_runs_per_question)
    else:
        summary += f"Skipped: Question type '{question_type}' is not yet supported.\n"
        print(summary)
        return summary

    summary += f"Forecast: {forecast}\nComment:\n{comment[:200]}...\n"
    
    if submit_prediction:
        try:
            payload = create_forecast_payload(forecast, question_type)
            post_question_prediction(question_id, payload)
            post_question_comment(post_id, comment)
            summary += "SUCCESS: Prediction and comment submitted.\n"
        except Exception as e:
            summary += f"ERROR: Failed to submit prediction. {e}\n"
    else:
        summary += "SKIPPED SUBMISSION: SUBMIT_PREDICTION is False.\n"
        
    print(summary)
    return summary

async def main():
    print("--- Starting Forecasting Bot ---")
    if not METACULUS_TOKEN:
        raise ValueError("METACULUS_TOKEN environment variable not set.")

    if USE_EXAMPLE_QUESTIONS:
        question_ids = EXAMPLE_QUESTIONS
    else:
        question_ids = get_open_question_ids_from_tournament()
    
    if not question_ids:
        print("No open questions found.")
        return

    tasks = [
        forecast_individual_question(
            q_id, p_id, SUBMIT_PREDICTION, NUM_RUNS_PER_QUESTION, SKIP_PREVIOUSLY_FORECASTED_QUESTIONS
        )
        for q_id, p_id in question_ids
    ]
    
    await asyncio.gather(*tasks)
    print("--- Forecasting Bot finished. ---")


if __name__ == "__main__":
    asyncio.run(main())
