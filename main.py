# A forecasting bot combining a structured framework with custom research tools
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

# -----------------------------
# Environment & API Keys
# -----------------------------
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")

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
    This bot integrates custom research functions (web scraping, NewsAPI)
    into the robust `forecasting-tools` framework.
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY) if NEWSAPI_API_KEY else None

    # --- Custom Research Implementation ---

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Orchestrates research calls using web scraping and NewsAPI.
        """
        async with self._concurrency_limiter:
            logger.info(f"--- Running Research for: {question.question_text} ---")
            
            # Run blocking I/O calls in a separate thread to avoid blocking the asyncio event loop
            loop = asyncio.get_running_loop()
            scrape_results = await loop.run_in_executor(None, self.perform_web_scrape, question.question_text)
            news_results = await loop.run_in_executor(None, self.call_newsapi, question.question_text)

            research = f"Web Scrape Summary:\n{scrape_results}\n\nRecent News:\n{news_results}"
            logger.info(f"--- Research Complete for Q {question.id} ---\n{research[:400]}...\n--------------------")
            return research

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

