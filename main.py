"""
This is the main entry point for the forecasting bot, combining a structured
framework with custom research tools.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Literal

import requests
from bs4 import BeautifulSoup
from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)
from newsapi import NewsApiClient

# --- Environment Variables ---
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")

# --- Logging Setup ---
logger = logging.getLogger(__name__)

class FrameworkBot(ForecastBot):
    """
    A forecasting bot for Metaculus tournaments that uses a structured framework
    with web scraping and NewsAPI for research.
    """
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Conducts research on a given question using direct web scraping and NewsAPI.
        """
        async with self._concurrency_limiter:
            logger.info(f"Starting research for: {question.question_text}")

            scrape_results = self.perform_web_scrape(question.question_text)
            news_results = self.call_newsapi(question.question_text)

            research_summary = f"Web Scrape Results:\n{scrape_results}\n\nRecent News Articles:\n{news_results}"
            
            logger.info(f"Research for URL {question.page_url}:\n{research_summary[:500]}...")
            return research_summary

    def perform_web_scrape(self, query: str) -> str:
        """Performs a web search and scrapes the top result."""
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
            page_response = requests.get(page_url, headers=headers, timeout=10)
            page_response.raise_for_status()
            
            page_soup = BeautifulSoup(page_response.text, 'html.parser')
            paragraphs = page_soup.find_all('p')
            content = ' '.join([p.get_text() for p in paragraphs[:5]])
            return f"- {content[:1500]}..."
        except Exception as e:
            return f"Web scraping failed: {e}"

    def call_newsapi(self, query: str) -> str:
        """Fetches recent news articles using the NewsAPI."""
        if not NEWSAPI_API_KEY:
            return "NewsAPI key not set."
        try:
            newsapi = NewsApiClient(api_key=NEWSAPI_API_KEY)
            articles = newsapi.get_everything(q=query, language="en", sort_by="relevancy", page_size=5)
            return "\n".join([f"- {article['title']}: {article.get('description', 'N/A')}" for article in articles.get("articles", [])])
        except Exception as e:
            return f"NewsAPI search failed: {e}"

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster. Your task is to predict the outcome of the following binary question.

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Research Summary: {research}
            Today's Date: {datetime.now().strftime("%Y-%m-%d")}

            First, provide a step-by-step reasoning for your forecast.
            Finally, state your final probability as a percentage: "Probability: ZZ%"
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

    # Placeholder methods for other question types
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        # This would be implemented similarly to the binary method
        logger.warning(f"Multiple choice forecasting not implemented for {question.page_url}. Skipping.")
        # Return a dummy prediction to avoid errors
        dummy_prediction = PredictedOptionList.from_probabilities({opt: 1/len(question.options) for opt in question.options})
        return ReasonedPrediction(prediction_value=dummy_prediction, reasoning="Not implemented.")

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        # This would be implemented similarly to the binary method
        logger.warning(f"Numeric forecasting not implemented for {question.page_url}. Skipping.")
        # Return a dummy prediction to avoid errors
        dummy_prediction = NumericDistribution.from_question([Percentile(percentile=50, value=question.lower_bound)], question)
        return ReasonedPrediction(prediction_value=dummy_prediction, reasoning="Not implemented.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress noisy logs from underlying libraries if needed
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Run the FrameworkBot forecasting system")
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
    framework_bot = FrameworkBot(
        research_reports_per_question=1,
        predictions_per_research_report=3,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(model="meta-llama/llama-3-8b-instruct:free"),
            "parser": GeneralLlm(model="openai/gpt-4o-mini"),
            "summarizer": GeneralLlm(model="google/gemini-1.5-flash-latest"),
        },
    )

    try:
        if run_mode == "tournament":
            forecast_reports = asyncio.run(
                framework_bot.forecast_on_tournament(
                    MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
                )
            )
        elif run_mode == "test_questions":
            EXAMPLE_URLS = [
                "https://www.metaculus.com/questions/578/human-extinction-by-2025/",
            ]
            framework_bot.skip_previously_forecasted_questions = False
            questions = [MetaculusApi.get_question_by_url(url) for url in EXAMPLE_URLS]
            forecast_reports = asyncio.run(
                framework_bot.forecast_questions(questions, return_exceptions=True)
            )
        
        framework_bot.log_report_summary(forecast_reports)
        logger.info("Run finished successfully.")
    except Exception as e:
        logger.error(f"Run failed with error: {e}", exc_info=True)

