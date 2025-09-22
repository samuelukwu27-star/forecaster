# A forecasting bot combining a structured framework with custom research tools    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Orchestrates research calls using web scraping, NewsAPI, and Tavily.
        """
        async with self._concurrency_limiter:
            logger.info(f"--- Running Research for: {question.question_text} ---")

            # Run blocking I/O calls in a separate thread to avoid blocking the asyncio event loop
            loop = asyncio.get_running_loop()
            tasks = {
                "tavily": loop.run_in_executor(None, self.call_tavily, question.question_text),
                "web_scrape": loop.run_in_executor(None, self.perform_web_scrape, question.question_text),
                "news": loop.run_in_executor(None, self.call_newsapi, question.question_text),
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            tavily_results, scrape_results, news_results = results[0], results[1], results[2]

            research = f"Tavily Research Summary:\n{tavily_results}\n\nWeb Scrape Summary:\n{scrape_results}\n\nRecent News:\n{news_results}"
            # FIX: Use page_url instead of id for logging, as 'id' is not a guaranteed attribute.
            logger.info(f"--- Research Complete for Q {question.page_url} ---\n{research[:400]}...\n--------------------")
            return research

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



# using top OpenRouter models for reasoning.

import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

import requests
from bs4 import BeautifulSoup
from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)
from newsapi import NewsApiClient
from tavily import TavilyClient

# -----------------------------
# Environment & API Keys
# -----------------------------
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("UnifiedForecastBot")


class UnifiedForecastingBot(ForecastBot):
    """
    This bot integrates custom research functions (web scraping, NewsAPI, Tavily)
    into the robust `forecasting-tools` framework.
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY) if NEWSAPI_API_KEY else None
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

    # --- Custom Research Implementation ---

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Orchestrates research calls using Tavily, web scraping, and NewsAPI.
        """
        async with self._concurrency_limiter:
            logger.info(f"--- Running Research for: {question.question_text} ---")
            
            loop = asyncio.get_running_loop()
            
            # Run all blocking I/O calls concurrently
            tasks = [
                loop.run_in_executor(None, self.call_tavily, question.question_text),
                loop.run_in_executor(None, self.perform_web_scrape, question.question_text),
                loop.run_in_executor(None, self.call_newsapi, question.question_text),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            tavily_results = results[0] if not isinstance(results[0], Exception) else f"Tavily search failed: {results[0]}"
            scrape_results = results[1] if not isinstance(results[1], Exception) else f"Web scraping failed: {results[1]}"
            news_results = results[2] if not isinstance(results[2], Exception) else f"NewsAPI search failed: {results[2]}"

            research = f"{tavily_results}\n\nWeb Scrape Summary:\n{scrape_results}\n\nRecent News:\n{news_results}"
            logger.info(f"--- Research Complete for Q {question.id} ---\n{research[:400]}...\n--------------------")
            return research

    def call_tavily(self, query: str) -> str:
        """Fetches research from Tavily."""
        if not self.tavily_client:
            logger.warning("[Tavily] Tavily API key not set. Skipping Tavily search.")
            return "Tavily search not performed."
        try:
            logger.info(f"[Tavily] Searching for: {query}")
            # Use search_depth="advanced" for more comprehensive results
            response = self.tavily_client.search(query=query, search_depth="advanced")
            # Format the response to be a concise summary
            context = "\n".join([f"- {obj['content']}" for obj in response.get('results', [])[:3]])
            return f"Tavily Research Summary:\n{context}"
        except Exception as e:
            return f"Tavily search failed: {e}"

    def perform_web_scrape(self, query: str) -> str:
        """Performs a web search using DuckDuckGo and scrapes the top result."""
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
            return f"- {content[:1500]}..."
        except Exception as e:
            return f"Web scraping failed: {e}"

    def call_newsapi(self, query: str) -> str:
        """Fetches recent news articles from NewsAPI."""
        if not self.newsapi_client:
            logger.warning("[NewsAPI] NewsAPI key not set. Skipping news search.")
            return "NewsAPI search not performed."
        try:
            all_articles = self.newsapi_client.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)
            if not all_articles or not all_articles['articles']:
                return "No recent news articles found."
            formatted_articles = ""
            for article in all_articles['articles']:
                formatted_articles += f"- Title: {article['title']}\n  Snippet: {article.get('description', 'N/A')}\n"
            return formatted_articles
        except Exception as e:
            return f"NewsAPI search failed: {e}"

    # --- Forecasting Prompts & Logic (from Fall 2025 Template) ---

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.
            Your interview question is:
            {question.question_text}
            Question background:
            {question.background_info}
            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}
            {question.fine_print}
            Your research assistant says:
            {research}
            Today is {datetime.now().strftime("%Y-%m-%d")}.
            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.
            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}"
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.
            Your interview question is:
            {question.question_text}
            The options are: {question.options}
            Background:
            {question.background_info}
            {question.resolution_criteria}
            {question.fine_print}
            Your research assistant says:
            {research}
            Today is {datetime.now().strftime("%Y-%m-%d")}.
            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.
            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}"
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.
            Your interview question is:
            {question.question_text}
            Background:
            {question.background_info}
            {question.resolution_criteria}
            {question.fine_print}
            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}
            Your research assistant says:
            {research}
            Today is {datetime.now().strftime("%Y-%m-%d")}.
            {lower_bound_message}
            {upper_bound_message}
            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there
            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.
            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.nominal_upper_bound is not None:
            upper_bound_number = question.nominal_upper_bound
        else:
            upper_bound_number = question.upper_bound
        if question.nominal_lower_bound is not None:
            lower_bound_number = question.nominal_lower_bound
        else:
            lower_bound_number = question.lower_bound

        upper_bound_message = (
            f"The question creator thinks the number is likely not higher than {upper_bound_number}."
            if question.open_upper_bound
            else f"The outcome can not be higher than {upper_bound_number}."
        )
        lower_bound_message = (
            f"The question creator thinks the number is likely not lower than {lower_bound_number}."
            if question.open_lower_bound
            else f"The outcome can not be lower than {lower_bound_number}."
        )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    # Suppress noisy logs from underlying libraries
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Run the UnifiedForecastingBot"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "test_questions"] = args.mode

    # Configure the bot with specific OpenRouter models for different tasks
    unified_bot = UnifiedForecastingBot(
        research_reports_per_question=1,
        predictions_per_research_report=3,  # This will result in 3 total LLM calls per question
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(model="openai/gpt-4o", temperature=0.3, timeout=40, allowed_tries=2),
            "parser": GeneralLlm(model="openai/gpt-4o-mini"),
            "summarizer": GeneralLlm(model="openai/gpt-4o-mini"),
            "researcher": GeneralLlm(model="openrouter/openai/gpt-4o-search-preview"),
        },
    )

    forecast_reports = []
    try:
        if run_mode == "tournament":
            logger.info("Running in tournament mode...")
            # Forecast on the main AI competition
            seasonal_reports = asyncio.run(
                unified_bot.forecast_on_tournament(
                    MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
                )
            )
            # Forecast on the MiniBench competition
            minibench_reports = asyncio.run(
                unified_bot.forecast_on_tournament(
                    MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
                )
            )
            forecast_reports = seasonal_reports + minibench_reports

        elif run_mode == "test_questions":
            logger.info("Running in test questions mode...")
            EXAMPLE_QUESTIONS = [
                "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Binary
                "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Numeric
                "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Multiple Choice
            ]
            unified_bot.skip_previously_forecasted_questions = False
            questions = [
                MetaculusApi.get_question_by_url(question_url)
                for question_url in EXAMPLE_QUESTIONS
            ]
            forecast_reports = asyncio.run(
                unified_bot.forecast_questions(questions, return_exceptions=True)
            )

        unified_bot.log_report_summary(forecast_reports)
        logger.info("Run finished successfully.")

    except Exception as e:
        logger.error(f"Run failed with a critical error: {e}", exc_info=True)

