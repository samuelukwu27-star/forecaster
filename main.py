# A forecasting bot using the forecasting-tools framework and custom research.
import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

import numpy as np
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
    into the robust `forecasting-tools` framework. It uses a multi-model approach
    for generating predictions.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    # FIX: Change from @property to a regular method as expected by the framework.
    def _llm_config_defaults(self) -> dict[str, str]:
        """
        Registers custom forecaster roles to suppress warnings.
        """
        defaults = super()._llm_config_defaults() # Call the parent method
        defaults.update({
            "forecaster_1": "openai/gpt-4o-mini",
            "forecaster_2": "perplexity/llama-3-sonar-large-32k-online",
            "forecaster_3": "anthropic/claude-3-haiku-20240307",
        })
        return defaults

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY)
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.forecaster_keys = ["forecaster_1", "forecaster_2", "forecaster_3"]

    # --- Custom Research Implementation ---

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Orchestrates research calls using web scraping, NewsAPI, and Tavily.
        """
        async with self._concurrency_limiter:
            logger.info(f"--- Running Research for: {question.question_text} ---")
            
            loop = asyncio.get_running_loop()
            tasks = {
                "tavily": loop.run_in_executor(None, self.call_tavily, question.question_text),
                "web_scrape": loop.run_in_executor(None, self.perform_web_scrape, question.question_text),
                "news": loop.run_in_executor(None, self.call_newsapi, question.question_text),
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            tavily_results, scrape_results, news_results = results[0], results[1], results[2]

            research = f"Tavily Research Summary:\n{tavily_results}\n\nWeb Scrape Summary:\n{scrape_results}\n\nRecent News:\n{news_results}"
            logger.info(f"--- Research Complete for Q {question.page_url} ---\n{research[:400]}...\n--------------------")
            return research

    def call_tavily(self, query: str) -> str:
        """Performs a research search using Tavily."""
        if not self.tavily_client.api_key:
            logger.warning("[Tavily] TAVILY_API_KEY not set. Skipping.")
            return "Tavily search not performed."
        try:
            logger.info(f"[Tavily] Searching for: {query}")
            response = self.tavily_client.search(query=query, search_depth="advanced")
            context = "\n".join([f"- {c['content']}" for c in response['results']])
            return f"Tavily Research Summary:\n{context[:2000]}"
        except Exception as e:
            return f"Tavily search failed: {e}"

    def perform_web_scrape(self, query: str) -> str:
        """Performs a web search and scrapes the top result."""
        logger.info(f"[Web Scraper] Searching for: {query}")
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            headers = {'User-Agent': 'Mozilla/5.0'}
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
        if not self.newsapi_client.api_key:
            logger.warning("[NewsAPI] NewsAPI key not set. Skipping.")
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

    # --- Multi-Model Forecasting Logic ---

    async def _get_reasonings_from_all_models(self, prompt: str) -> list[str]:
        """Invokes all configured forecaster models with the same prompt."""
        tasks = [
            self.get_llm(key, "llm").invoke(prompt) for key in self.forecaster_keys
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _combine_reasonings_for_comment(self, reasonings: list) -> str:
        """Formats reasonings from all models into a single string for submission."""
        comment = ""
        for i, reasoning in enumerate(reasonings):
            model_key = self.forecaster_keys[i]
            model_name = self.get_llm(model_key, "model_name")
            comment += f"--- Reasoning from Model: {model_name} ---\n\n"
            if isinstance(reasoning, Exception):
                comment += f"ERROR: {reasoning}\n\n"
            else:
                comment += f"{reasoning}\n\n"
        return comment

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster. Your interview question is:
            {question.question_text}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            {question.fine_print}
            Your research assistant says:
            {research}
            Today is {datetime.now().strftime("%Y-%m-%d")}.
            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A scenario that results in a No outcome.
            (d) A scenario that results in a Yes outcome.
            You write your rationale, putting extra weight on the status quo.
            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        
        reasonings = await self._get_reasonings_from_all_models(prompt)
        
        parsing_tasks = [
            structure_output(r, BinaryPrediction, model=self.get_llm("parser", "llm"))
            for r in reasonings if not isinstance(r, Exception)
        ]
        predictions = await asyncio.gather(*parsing_tasks, return_exceptions=True)
        
        valid_preds = [
            p.prediction_in_decimal 
            for p in predictions if not isinstance(p, Exception)
        ]
        
        if not valid_preds:
            raise ValueError("All model predictions failed parsing.")

        median_pred = float(np.median(valid_preds))
        final_pred = max(0.01, min(0.99, median_pred))
        
        combined_reasoning = self._combine_reasonings_for_comment(reasonings)
        
        logger.info(f"Forecasted {question.page_url} with median prediction: {final_pred}")
        return ReasonedPrediction(prediction_value=final_pred, reasoning=combined_reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster. Your question is:
            {question.question_text}
            Options: {question.options}
            Background: {question.background_info}
            {question.resolution_criteria}
            Research:
            {research}
            Today is {datetime.now().strftime("%Y-%m-%d")}.
            Write your rationale, considering the status quo and unexpected outcomes.
            The last thing you write is your final probabilities for the options {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            """
        )
        parsing_instructions = f"Ensure option names are one of: {question.options}"

        reasonings = await self._get_reasonings_from_all_models(prompt)

        parsing_tasks = [
            structure_output(
                r, PredictedOptionList, self.get_llm("parser", "llm"),
                additional_instructions=parsing_instructions
            ) for r in reasonings if not isinstance(r, Exception)
        ]
        predictions = await asyncio.gather(*parsing_tasks, return_exceptions=True)

        valid_preds = [p for p in predictions if not isinstance(p, Exception)]
        if not valid_preds:
            raise ValueError("All model predictions failed parsing.")

        avg_probs = {}
        for option in question.options:
            option_probs = [p.get_prob(option) for p in valid_preds]
            avg_probs[option] = np.mean(option_probs)
        
        total_prob = sum(avg_probs.values())
        final_probs = {option: prob / total_prob for option, prob in avg_probs.items()}

        final_prediction = PredictedOptionList(list(final_probs.items()))
        combined_reasoning = self._combine_reasonings_for_comment(reasonings)

        logger.info(f"Forecasted {question.page_url} with prediction: {final_prediction}")
        return ReasonedPrediction(prediction_value=final_prediction, reasoning=combined_reasoning)


    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster. Your question is:
            {question.question_text}
            Background: {question.background_info}
            Units for answer: {question.unit_of_measure or "Not stated"}
            Research:
            {research}
            Today is {datetime.now().strftime("%Y-%m-%d")}.
            {lower_bound_message}
            {upper_bound_message}
            Write your rationale, considering expert expectations and unexpected scenarios.
            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 50: XX
            Percentile 90: XX
            "
            """
        )
        
        reasonings = await self._get_reasonings_from_all_models(prompt)

        parsing_tasks = [
            structure_output(r, list[Percentile], self.get_llm("parser", "llm"))
            for r in reasonings if not isinstance(r, Exception)
        ]
        predictions = await asyncio.gather(*parsing_tasks, return_exceptions=True)

        valid_preds = [p for p in predictions if not isinstance(p, Exception)]
        if not valid_preds:
            raise ValueError("All model predictions failed parsing.")
        
        median_percentiles = []
        percentile_levels = sorted({p.percentile for pred_list in valid_preds for p in pred_list})

        for level in percentile_levels:
            values = [p.value for pred_list in valid_preds for p in pred_list if p.percentile == level]
            if values:
                median_value = np.median(values)
                median_percentiles.append(Percentile(percentile=level, value=median_value))
        
        final_prediction = NumericDistribution.from_question(median_percentiles, question)
        combined_reasoning = self._combine_reasonings_for_comment(reasonings)

        logger.info(f"Forecasted {question.page_url} with prediction: {final_prediction.declared_percentiles}")
        return ReasonedPrediction(prediction_value=final_prediction, reasoning=combined_reasoning)


    def _create_upper_and_lower_bound_messages(self, question: NumericQuestion) -> tuple[str, str]:
        upper_bound = question.nominal_upper_bound or question.upper_bound
        lower_bound = question.nominal_lower_bound or question.lower_bound
        upper_msg = f"The number is likely not higher than {upper_bound}." if question.open_upper_bound else f"The outcome cannot be higher than {upper_bound}."
        lower_msg = f"The number is likely not lower than {lower_bound}." if question.open_lower_bound else f"The outcome cannot be lower than {lower_bound}."
        return upper_msg, lower_msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the UnifiedForecastingBot.")
    parser.add_argument(
        "--mode", type=str, choices=["tournament", "test_questions"],
        default="tournament", help="Specify the run mode.",
    )
    parser.add_argument(
        "--tournament-ids", nargs='+', type=str,
        help="One or more tournament IDs or slugs to forecast on (e.g., 32813 metaculus-cup)."
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "test_questions"] = args.mode

    unified_bot = UnifiedForecastingBot(
        research_reports_per_question=1,
        predictions_per_research_report=1, 
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(model="openai/gpt-4o-mini"),
            "summarizer": GeneralLlm(model="openai/gpt-4o-mini"),
            "researcher": None, 
            "forecaster_1": GeneralLlm(model="openai/gpt-4o-mini", temperature=0.3),
            "forecaster_2": GeneralLlm(model="perplexity/llama-3-sonar-large-32k-online", temperature=0.3),
            "forecaster_3": GeneralLlm(model="anthropic/claude-3-haiku-20240307", temperature=0.3),
            "parser": GeneralLlm(model="openai/gpt-4o-mini"),
        },
    )

    forecast_reports = []
    try:
        if run_mode == "tournament":
            logger.info("Running in tournament mode...")
            tournament_ids_to_run = args.tournament_ids or [
                MetaculusApi.CURRENT_AI_COMPETITION_ID,
                MetaculusApi.CURRENT_MINIBENCH_ID
            ]
            logger.info(f"Targeting tournaments: {tournament_ids_to_run}")

            all_reports = []
            for tournament_id in tournament_ids_to_run:
                reports = asyncio.run(
                    unified_bot.forecast_on_tournament(
                        tournament_id, return_exceptions=True
                    )
                )
                all_reports.extend(reports)
            forecast_reports = all_reports

        elif run_mode == "test_questions":
            logger.info("Running in test questions mode...")
            EXAMPLE_QUESTIONS = [
                "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
                "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
                "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            ]
            questions = [MetaculusApi.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
            forecast_reports = asyncio.run(
                unified_bot.forecast_questions(questions, return_exceptions=True)
            )

        unified_bot.log_report_summary(forecast_reports)
        logger.info("Run finished successfully.")

    except Exception as e:
        logger.error(f"Run failed with a critical error: {e}", exc_info=True)

