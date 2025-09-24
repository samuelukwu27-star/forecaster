import argparse
import asyncio
import logging
import os
import re
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
    This bot uses a committee of independent synthesizer models to forecast on
    binary, numeric, and multiple-choice questions. The reasoning structure is
    adapted for each question type (e.g., proponent/opponent for binary,
    high/low analysts for numeric).
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        """
        Registers custom agent roles to suppress warnings and provide sane defaults.
        All models are explicitly routed through OpenRouter.
        """
        defaults = super()._llm_config_defaults()
        # FIX: Prepend "openrouter/" to all model names to force routing
        defaults.update({
            "proponent": "openrouter/openai/gpt-4o",
            "opponent": "openrouter/openai/gpt-4-turbo",
            "analyst_low": "openrouter/openai/gpt-4o",
            "analyst_high": "openrouter/openai/gpt-4-turbo",
            "analyst_mc": "openrouter/openai/gpt-4o",
            "synthesizer_1": "openrouter/openai/gpt-4o",
            "synthesizer_2": "openrouter/openai/gpt-4-turbo",
            "synthesizer_3": "openrouter/openai/gpt-4o",
        })
        return defaults

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY)
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.synthesizer_keys = [k for k in self._llms.keys() if k.startswith("synthesizer")]
        if not self.synthesizer_keys:
            raise ValueError("No synthesizer models found. Define at least one 'synthesizer_1'.")
        logger.info(f"Initialized with a committee of {len(self.synthesizer_keys)} synthesizers.")

    # --- Research Implementation ---
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            logger.info(f"--- Running Research for: {question.question_text} ---")
            loop = asyncio.get_running_loop()
            tasks = {
                "tavily": loop.run_in_executor(None, self.call_tavily, question.question_text),
                "news": loop.run_in_executor(None, self.call_newsapi, question.question_text),
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            tavily_response, news_results = results

            tavily_summary = "Tavily search failed."
            urls_to_scrape = []
            if isinstance(tavily_response, dict):
                tavily_summary = "\n".join([f"- {c['content']}" for c in tavily_response.get('results', [])])
                urls_to_scrape = [c['url'] for c in tavily_response.get('results', [])][:3]

            scraped_data = ""
            if urls_to_scrape:
                logger.info(f"Scraping {len(urls_to_scrape)} URLs...")
                scraped_data = await loop.run_in_executor(None, self.scrape_urls, urls_to_scrape)

            raw_research = (
                f"Tavily Summary:\n{tavily_summary}\n\nRecent News:\n{news_results}\n\n"
                f"Web Scraped Content:\n{scraped_data}"
            )

            logger.info(f"--- Synthesizing Raw Research for: {question.question_text} ---")
            synthesis_prompt = clean_indents(f"""
                Analyze the following raw research data. Provide a concise, synthesized summary for a forecaster.
                Focus on key drivers, potential turning points, and conflicting information.
                Raw Data:\n{raw_research}\n\nSynthesized Summary:
            """)
            synthesized_research = await self.get_llm("researcher", "llm").invoke(synthesis_prompt)
            logger.info(f"--- Research Complete for Q {question.page_url} ---")
            return synthesized_research

    def scrape_urls(self, urls: list[str]) -> str:
        scraped_content = []
        for url in urls:
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(url, timeout=10, headers=headers)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    page_text = ' '.join([p.get_text() for p in soup.find_all('p')])
                    scraped_content.append(f"--- From {url} ---\n{page_text.strip()[:2500]}...")
                else:
                    logger.warning(f"Failed to fetch {url}, status: {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
        return "\n\n".join(scraped_content)

    def call_tavily(self, query: str) -> dict | str:
        if not self.tavily_client.api_key: return "Tavily key not set."
        try:
            return self.tavily_client.search(query=query, search_depth="advanced", max_results=5)
        except Exception as e:
            logger.error(f"Tavily search failed: {e}"); return f"Tavily search failed: {e}"

    def call_newsapi(self, query: str) -> str:
        if not self.newsapi_client.api_key: return "NewsAPI key not set."
        try:
            articles = self.newsapi_client.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)
            if not articles or not articles.get('articles'): return "No recent news found."
            return "\n".join([f"- {a['title']}: {a.get('description', 'N/A')}" for a in articles['articles']])
        except Exception as e:
            logger.error(f"NewsAPI search failed: {e}"); return f"NewsAPI search failed: {e}"

    # --- Forecasting Logic ---
    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        logger.info(f"--- Starting Binary Debate for: {question.page_url} ---")
        proponent_arg = await self.get_llm("proponent", "llm").invoke(clean_indents(f"""
            Act as a PROPONENT for a YES outcome. Build the strongest possible case using the research.
            Question: {question.question_text}, Research: {research}
        """))
        opponent_arg = await self.get_llm("opponent", "llm").invoke(clean_indents(f"""
            Act as an OPPONENT for a NO outcome. Build the strongest possible case using the research.
            Question: {question.question_text}, Research: {research}
        """))

        synthesizer_prompt = clean_indents(f"""
            You are a judge on a forecasting committee. Evaluate the competing arguments to arrive at a final probability.
            Question: "{question.question_text}"
            Resolution Criteria: {question.resolution_criteria}, Research: {research}
            --- Proponent's Case for YES ---\n{proponent_arg}\n--- END CASE ---
            --- Opponent's Case for NO ---\n{opponent_arg}\n--- END CASE ---
            1. Summarize the strongest point from each side.
            2. Identify gaps or weaknesses in their arguments.
            3. Write your final integrated rationale.
            4. The very last thing you write is your final probability as: "Probability: ZZ%", from 0-100.
        """)

        predictions = await self._run_committee_and_parse(synthesizer_prompt, BinaryPrediction)
        valid_preds = [p.prediction_in_decimal for p in predictions if p and hasattr(p, "prediction_in_decimal")]
        if not valid_preds: raise ValueError("All binary synthesizer predictions failed parsing.")

        median_pred = float(np.median(valid_preds))
        final_pred = max(0.01, min(0.99, median_pred))
        comment = self._format_comment("Debate", {"Proponent": proponent_arg, "Opponent": opponent_arg})

        logger.info(f"Binary forecast for {question.page_url}: {final_pred:.2%}")
        return ReasonedPrediction(prediction_value=final_pred, reasoning=comment)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        logger.info(f"--- Starting Numeric Analysis for: {question.page_url} ---")
        low_arg = await self.get_llm("analyst_low", "llm").invoke(clean_indents(f"""
            Act as an analyst arguing for a LOWER estimate. Build a case for why the final number will be on the low end of the plausible range.
            Question: {question.question_text}, Research: {research}
        """))
        high_arg = await self.get_llm("analyst_high", "llm").invoke(clean_indents(f"""
            Act as an analyst arguing for a HIGHER estimate. Build a case for why the final number will be on the high end of the plausible range.
            Question: {question.question_text}, Research: {research}
        """))

        synthesizer_prompt = clean_indents(f"""
            You are a judge on a forecasting committee. Evaluate the competing analyses to arrive at a numeric distribution.
            Question: "{question.question_text}"
            Resolution Criteria: {question.resolution_criteria}, Research: {research}
            --- Analyst Case for a LOW number ---\n{low_arg}\n--- END CASE ---
            --- Analyst Case for a HIGH number ---\n{high_arg}\n--- END CASE ---
            1. Summarize the strongest point from each analyst.
            2. Write your final integrated rationale.
            3. The very last thing you write must be your final distribution as three percentiles:
            "P25: [value]\nP50: [value]\nP75: [value]"
        """)

        predictions = await self._run_committee_and_parse(synthesizer_prompt, NumericDistribution)
        valid_preds = [p for p in predictions if p]
        if not valid_preds: raise ValueError("All numeric synthesizer predictions failed parsing.")

        p25 = np.median([p.p25 for p in valid_preds])
        p50 = np.median([p.p50 for p in valid_preds])
        p75 = np.median([p.p75 for p in valid_preds])
        final_dist = NumericDistribution(p25=float(p25), p50=float(p50), p75=float(p75))
        comment = self._format_comment("Numeric Analysis", {"Analyst Low": low_arg, "Analyst High": high_arg})

        logger.info(f"Numeric forecast for {question.page_url}: {final_dist}")
        return ReasonedPrediction(prediction_value=final_dist, reasoning=comment)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        logger.info(f"--- Starting Multi-Option Analysis for: {question.page_url} ---")
        options_str = "\n".join([f"- {opt}" for opt in question.options])
        analyst_arg = await self.get_llm("analyst_mc", "llm").invoke(clean_indents(f"""
            Act as an analyst. Provide a detailed evaluation of each possible option for the question based on the provided research.
            Question: {question.question_text}
            Options:\n{options_str}\nResearch: {research}
        """))

        synthesizer_prompt = clean_indents(f"""
            You are a judge on a forecasting committee. Use the analyst's report and research to assign probabilities to each option.
            Question: "{question.question_text}"
            Options:\n{options_str}\nResearch: {research}
            --- Analyst's Evaluation of Options ---\n{analyst_arg}\n--- END EVALUATION ---
            1. Write your final integrated rationale.
            2. The very last thing you write is your list of probabilities. The probabilities MUST sum to 100%. Format it exactly as:
            "Predictions:
            Option Name 1: XX%
            Option Name 2: YY%
            ..."
        """)
        predictions = await self._run_committee_and_parse(synthesizer_prompt, PredictedOptionList, question.options)
        valid_preds = [p.as_dict for p in predictions if p]
        if not valid_preds: raise ValueError("All multi-choice synthesizer predictions failed parsing.")

        avg_probs = {option: np.mean([pred[option] for pred in valid_preds]) for option in question.options}
        total_prob = sum(avg_probs.values())
        final_probs = {option: prob / total_prob for option, prob in avg_probs.items()}
        final_pred = PredictedOptionList(list(final_probs.items()))
        comment = self._format_comment("Multi-Option Analysis", {"Analyst": analyst_arg})

        logger.info(f"Multi-choice forecast for {question.page_url}: {final_pred}")
        return ReasonedPrediction(prediction_value=final_pred, reasoning=comment)

    async def _run_committee_and_parse(self, prompt: str, output_type, options=None):
        logger.info(f"Presenting to committee of {len(self.synthesizer_keys)}...")
        tasks = [self.get_llm(key, "llm").invoke(prompt) for key in self.synthesizer_keys]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        parsing_tasks = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Synthesizer failed: {res}")
                parsing_tasks.append(asyncio.sleep(0, result=None))
                continue
            
            if output_type == PredictedOptionList:
                 parsing_tasks.append(structure_output(res, output_type, self.get_llm("parser", "llm"), options=options))
            else:
                 parsing_tasks.append(structure_output(res, output_type, self.get_llm("parser", "llm")))

        parsed = await asyncio.gather(*parsing_tasks, return_exceptions=True)
        return [p for p in parsed if not isinstance(p, Exception)]

    def _format_comment(self, stage_name: str, args: dict) -> str:
        comment = f"--- {stage_name.upper()} STAGE ---\n\n"
        for agent_name, reasoning in args.items():
            model_key = agent_name.lower().replace(" ", "_")
            model_id = getattr(self.get_llm(model_key, 'llm'), 'model', 'unknown-model')
            comment += f"--- Argument from {agent_name} Agent ({model_id}) ---\n\n{reasoning}\n\n"
        return comment

async def main():
    parser = argparse.ArgumentParser(description="Run the CommitteeForecastingBot.")
    parser.add_argument("--mode", type=str, choices=["tournament", "test_questions"], default="tournament")
    parser.add_argument("--tournament-ids", nargs='+', type=str)
    args = parser.parse_args()

    committee_bot = CommitteeForecastingBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        llms={
            # FIX: Prepend "openrouter/" to all model names to force routing
            "default": GeneralLlm(model="openrouter/openai/gpt-4o-mini"),
            "summarizer": GeneralLlm(model="openrouter/openai/gpt-4o-mini"),
            "researcher": GeneralLlm(model="openrouter/openai/gpt-4o", temperature=0.1),
            "parser": GeneralLlm(model="openrouter/openai/gpt-4o"),
            "proponent": GeneralLlm(model="openrouter/openai/gpt-4o", temperature=0.4),
            "opponent": GeneralLlm(model="openrouter/openai/gpt-4-turbo", temperature=0.4),
            "analyst_low": GeneralLlm(model="openrouter/openai/gpt-4o", temperature=0.4),
            "analyst_high": GeneralLlm(model="openrouter/openai/gpt-4-turbo", temperature=0.4),
            "analyst_mc": GeneralLlm(model="openrouter/openai/gpt-4o", temperature=0.3),
            "synthesizer_1": GeneralLlm(model="openrouter/openai/gpt-4o", temperature=0.2),
            "synthesizer_2": GeneralLlm(model="openrouter/openai/gpt-4-turbo", temperature=0.2),
            "synthesizer_3": GeneralLlm(model="openrouter/openai/gpt-4o", temperature=0.2),
        },
    )

    forecast_reports = []
    try:
        if args.mode == "tournament":
            logger.info("Running in tournament mode...")
            ids = args.tournament_ids or [MetaculusApi.CURRENT_AI_COMPETITION_ID]
            logger.info(f"Targeting tournaments: {ids}")
            all_reports = []
            for tournament_id in ids:
                reports = await committee_bot.forecast_on_tournament(tournament_id, return_exceptions=True)
                all_reports.extend(reports)
            forecast_reports = all_reports
        else: # test_questions
            logger.info("Running in test questions mode...")
            URLS = [
                "https://www.metaculus.com/questions/578/human-extinction-by-2100/", # Binary
                "https://www.metaculus.com/questions/3475/date-of-first-human-on-mars/", # Numeric
                "https://www.metaculus.com/questions/2618/cause-of-next-human-extinction-event/" # Multiple Choice
            ]
            questions = [MetaculusApi.get_question_by_url(url) for url in URLS]
            forecast_reports = await committee_bot.forecast_questions(questions, return_exceptions=True)

        committee_bot.log_report_summary(forecast_reports)
        logger.info("Run finished successfully.")
    except Exception as e:
        logger.error(f"Run failed with a critical error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())


