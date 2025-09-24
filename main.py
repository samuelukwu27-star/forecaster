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
    AskNewsSearcher,
    SmartSearcher,
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
NEWSAPI_API_KEY = os.getenv("NEWSAPI_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CommitteeForecastBot")


class CommitteeForecastingBot(ForecastBot):
    """
    This bot uses a proponent/opponent debate structure, which is then evaluated
    by a "committee" of multiple, independent synthesizer models. The median
    prediction from the committee is used as the final forecast.
    It now incorporates web scraping of top search results for more in-depth research.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY)
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.synthesizer_keys = [k for k in self.llms.keys() if k.startswith("synthesizer")]
        if not self.synthesizer_keys:
            raise ValueError("No synthesizer models found in LLM configuration. Please define at least one 'synthesizer_1'.")
        logger.info(f"Initialized with a committee of {len(self.synthesizer_keys)} synthesizers.")

    # --- Custom Research Implementation ---
    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Orchestrates the research process:
        1. Calls Tavily and NewsAPI concurrently.
        2. Scrapes the top URLs returned by Tavily.
        3. Combines all information.
        4. Uses a researcher LLM to synthesize a final summary.
        """
        async with self._concurrency_limiter:
            logger.info(f"--- Running Raw Data Research for: {question.question_text} ---")
            loop = asyncio.get_running_loop()

            # Step 1: Run Tavily and NewsAPI in parallel
            tasks = {
                "tavily": loop.run_in_executor(None, self.call_tavily, question.question_text),
                "news": loop.run_in_executor(None, self.call_newsapi, question.question_text),
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            tavily_response, news_results = results[0], results[1]

            tavily_summary = "Tavily search failed."
            urls_to_scrape = []
            if isinstance(tavily_response, dict):
                tavily_summary = "\n".join([f"- {c['content']}" for c in tavily_response.get('results', [])])
                urls_to_scrape = [c['url'] for c in tavily_response.get('results', [])]

            # Step 2: Scrape URLs from Tavily results
            scraped_data = ""
            if urls_to_scrape:
                logger.info(f"Scraping {len(urls_to_scrape)} URLs from Tavily results...")
                scraped_data = await loop.run_in_executor(None, self.scrape_urls, urls_to_scrape)

            # Step 3: Combine all research sources
            raw_research = (
                f"Tavily Research Summary:\n{tavily_summary}\n\n"
                f"Recent News:\n{news_results}\n\n"
                f"--- Web Scraped Content ---\n{scraped_data}\n--- End Web Scraped Content ---"
            )

            # Step 4: Synthesize the combined research
            logger.info(f"--- Synthesizing Raw Research for: {question.question_text} ---")
            synthesis_prompt = clean_indents(f"""
                Analyze the following raw research data from Tavily, NewsAPI, and web scraping.
                Provide a concise, synthesized summary for a forecaster.
                Focus on the key drivers, potential turning points, and any conflicting information.

                Raw Data:
                {raw_research}

                Synthesized Summary:
            """)
            synthesized_research = await self.get_llm("researcher", "llm").invoke(synthesis_prompt)
            logger.info(f"--- Research Complete for Q {question.page_url} ---\n{synthesized_research[:400]}...\n--------------------")
            return synthesized_research

    def scrape_urls(self, urls: list[str]) -> str:
        """
        Scrapes the content of the given URLs using requests and BeautifulSoup.
        """
        scraped_content = []
        for url in urls:
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(url, timeout=10, headers=headers)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    paragraphs = soup.find_all('p')
                    page_text = ' '.join([p.get_text() for p in paragraphs])
                    # Limit text length to avoid excessive data for the synthesis model
                    page_text = page_text.strip()[:2500]
                    scraped_content.append(f"--- Content from {url} ---\n{page_text}...")
                else:
                    logger.warning(f"Failed to fetch {url} with status code {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
        return "\n\n".join(scraped_content)

    def call_tavily(self, query: str) -> dict | str:
        """
        Calls the Tavily API and returns the raw response dictionary on success.
        """
        if not self.tavily_client.api_key:
            logger.warning("Tavily API key not set. Skipping Tavily search.")
            return "Tavily search not performed (API key not set)."
        try:
            response = self.tavily_client.search(query=query, search_depth="advanced", max_results=5)
            return response
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return f"Tavily search failed: {e}"

    def call_newsapi(self, query: str) -> str:
        if not self.newsapi_client.api_key:
            logger.warning("NewsAPI key not set. Skipping NewsAPI search.")
            return "NewsAPI search not performed (API key not set)."
        try:
            articles = self.newsapi_client.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)
            if not articles or not articles.get('articles'):
                return "No recent news articles found."
            return "\n".join([f"- Title: {a['title']}\n  Snippet: {a.get('description', 'N/A')}" for a in articles['articles']])
        except Exception as e:
            logger.error(f"NewsAPI search failed: {e}")
            return f"NewsAPI search failed: {e}"

    # --- Committee Forecasting Logic ---
    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        logger.info(f"--- Starting Committee Debate for: {question.page_url} ---")
        today = datetime.now().strftime("%Y-%m-%d")

        proponent_prompt = clean_indents(f"""
            You are a professional superforecaster acting as a PROPONENT. Your goal is to build the strongest possible case for a YES outcome.
            Question: {question.question_text}, Background: {question.background_info}, Research: {research}, Today is {today}.
            Analyze the information and construct a persuasive argument for why the answer to the question will be YES.
            Start your response with your detailed rationale. Do not output a probability.
        """)
        proponent_argument = await self.get_llm("proponent", "llm").invoke(proponent_prompt)
        logger.info(f"Proponent argument generated for {question.page_url}")

        opponent_prompt = clean_indents(f"""
            You are a professional superforecaster acting as an OPPONENT. Your goal is to build the strongest possible case for a NO outcome.
            Question: {question.question_text}, Background: {question.background_info}, Research: {research}, Today is {today}.
            Analyze the information and construct a persuasive argument for why the answer to the question will be NO.
            Start your response with your detailed rationale. Do not output a probability.
        """)
        opponent_argument = await self.get_llm("opponent", "llm").invoke(opponent_prompt)
        logger.info(f"Opponent argument generated for {question.page_url}")

        synthesizer_prompt = clean_indents(f"""
            You are a professional superforecaster acting as a judge on a forecasting committee.
            Your task is to evaluate competing arguments to arrive at a final, precise probability.
            The question is: "{question.question_text}"
            Resolution Criteria: {question.resolution_criteria}
            Research Summary: {research}
            --- Proponent's Case for YES ---\n{proponent_argument}\n--- END OF PROPONENT'S CASE ---
            --- Opponent's Case for NO ---\n{opponent_argument}\n--- END OF OPPONENT'S CASE ---
            Today is {today}.
            Now, perform the following steps:
            1. Impartially summarize the strongest point from the proponent and the opponent.
            2. Identify any gaps or weaknesses in their arguments.
            3. Based on your evaluation, write your final integrated rationale.
            4. The very last thing you write is your final probability as: "Probability: ZZ%", from 0-100.
        """)

        logger.info(f"Presenting debate to the committee of {len(self.synthesizer_keys)} synthesizers...")
        tasks = [self.get_llm(key, "llm").invoke(synthesizer_prompt) for key in self.synthesizer_keys]
        synthesizer_reasonings_list = await asyncio.gather(*tasks, return_exceptions=True)
        synthesizer_reasonings_dict = dict(zip(self.synthesizer_keys, synthesizer_reasonings_list))

        logger.info("Parsing predictions from committee members...")
        parsing_tasks = [structure_output(r, BinaryPrediction, self.get_llm("parser", "llm")) for r in synthesizer_reasonings_list if not isinstance(r, Exception)]
        predictions = await asyncio.gather(*parsing_tasks, return_exceptions=True)
        valid_preds = [p.prediction_in_decimal for p in predictions if not isinstance(p, Exception) and hasattr(p, "prediction_in_decimal")]

        if not valid_preds:
            logger.error("All synthesizer predictions failed parsing.")
            raise ValueError("All synthesizer predictions failed parsing.")

        median_pred = float(np.median(valid_preds))
        final_pred = max(0.01, min(0.99, median_pred))

        combined_comment = self._format_committee_comment(proponent_argument, opponent_argument, synthesizer_reasonings_dict)

        logger.info(f"Forecasted {question.page_url} with committee median prediction: {final_pred} from {len(valid_preds)} valid predictions.")
        return ReasonedPrediction(prediction_value=final_pred, reasoning=combined_comment)

    def _format_committee_comment(self, proponent_arg: str, opponent_arg: str, synth_reasonings: dict) -> str:
        comment = "--- DEBATE STAGE ---\n\n"
        comment += f"--- Argument from Proponent Agent ({getattr(self.llms['proponent'], 'model', 'unknown-model')}) ---\n\n{proponent_arg}\n\n"
        comment += f"--- Argument from Opponent Agent ({getattr(self.llms['opponent'], 'model', 'unknown-model')}) ---\n\n{opponent_arg}\n\n"
        comment += "--- COMMITTEE EVALUATION STAGE ---\n\n"

        for agent_key, reasoning in synth_reasonings.items():
            model_name = getattr(self.llms[agent_key], 'model', "unknown-model")
            comment += f"--- Synthesizer Analysis from {agent_key} ({model_name}) ---\n\n"
            comment += f"ERROR: {reasoning}\n\n" if isinstance(reasoning, Exception) else f"{reasoning}\n\n"
        return comment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CommitteeForecastingBot.")
    parser.add_argument("--mode", type=str, choices=["tournament", "test_questions"], default="tournament")
    parser.add_argument("--tournament-ids", nargs='+', type=str)
    args = parser.parse_args()
    run_mode: Literal["tournament", "test_questions"] = args.mode

    committee_bot = CommitteeForecastingBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        llms={
            # General purpose models
            "default": GeneralLlm(model="openai/gpt-4o-mini"),
            "summarizer": GeneralLlm(model="openai/gpt-4o-mini"),
            "researcher": GeneralLlm(model="openai/gpt-4o", temperature=0.1),
            "parser": GeneralLlm(model="openai/gpt-4o"),

            # Debate agents using GPT-4o and GPT-4 Turbo
            "proponent": GeneralLlm(model="openai/gpt-4o", temperature=0.4),
            "opponent": GeneralLlm(model="openai/gpt-4-turbo", temperature=0.4),

            # Committee synthesizers using a mix of GPT-4o and GPT-4 Turbo
            "synthesizer_1": GeneralLlm(model="openai/gpt-4o", temperature=0.2),
            "synthesizer_2": GeneralLlm(model="openai/gpt-4-turbo", temperature=0.2),
            "synthesizer_3": GeneralLlm(model="openai/gpt-4o", temperature=0.2),
            "synthesizer_4": GeneralLlm(model="openai/gpt-4-turbo", temperature=0.2),
            "synthesizer_5": GeneralLlm(model="openai/gpt-4o", temperature=0.2),
        },
    )

    try:
        if run_mode == "tournament":
            logger.info("Running in tournament mode...")
            tournament_ids_to_run = args.tournament_ids or [MetaculusApi.CURRENT_AI_COMPETITION_ID]
            logger.info(f"Targeting tournaments: {tournament_ids_to_run}")
            all_reports = []
            for tournament_id in tournament_ids_to_run:
                reports = asyncio.run(committee_bot.forecast_on_tournament(tournament_id, return_exceptions=True))
                all_reports.extend(reports)
            forecast_reports = all_reports
        elif run_mode == "test_questions":
            logger.info("Running in test questions mode...")
            EXAMPLE_QUESTIONS = ["https://www.metaculus.com/questions/578/human-extinction-by-2100/"]
            questions = [MetaculusApi.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
            forecast_reports = asyncio.run(committee_bot.forecast_questions(questions, return_exceptions=True))

        committee_bot.log_report_summary(forecast_reports)
        logger.info("Run finished successfully.")
    except Exception as e:
        logger.error(f"Run failed with a critical error: {e}", exc_info=True)

