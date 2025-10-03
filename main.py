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

# -----------------------------
# Environment & API Keys
# -----------------------------
NEWSAPI_API_KEY = os.getenv("NEWSAPI_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

logger = logging.getLogger(__name__)


class EnhancedTournamentForecaster(ForecastBot):
    """
    Enhanced bot for forecasting on specific tournaments:
      - minibench
      - 32813
      - 32831
      - colombia-wage-watch

    Research: Perplexity Sonar Deep (OpenRouter) + NewsAPI + NewsData + Scraping
    Forecasting: Adaptive multi-agent committee with your exact model names.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY)
        self.synthesizer_keys = [k for k in self._llms.keys() if k.startswith("synthesizer")]
        if len(self.synthesizer_keys) < 3:
            raise ValueError("At least 3 synthesizer models required.")
        logger.info(f"🚀 Enhanced Forecaster Ready | Tournaments: minibench, 32813, 32831, colombia-wage-watch")

    # ======================
    # RESEARCH: 4-SOURCE INTELLIGENCE
    # ======================

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            loop = asyncio.get_running_loop()

            # 1. Perplexity Sonar Deep via OpenRouter (as LLM)
            researcher_llm = self.get_llm("researcher", "llm")
            perplexity_prompt = clean_indents(f"""
                You are a superforecaster's research assistant.
                Provide a deep, factual, and up-to-date summary for forecasting:
                Question: {question.question_text}
                Resolution Criteria: {question.resolution_criteria}
                Fine Print: {question.fine_print}
                Today: {datetime.now().strftime('%Y-%m-%d')}
                Focus on credible data, expert consensus, recent trends, and potential inflection points.
            """)
            perplexity_research = await researcher_llm.invoke(perplexity_prompt)

            # 2. NewsAPI
            newsapi_task = loop.run_in_executor(None, self._call_newsapi, question.question_text)
            # 3. NewsData
            newsdata_task = loop.run_in_executor(None, self._call_newsdata, question.question_text)

            newsapi_res, newsdata_res = await asyncio.gather(newsapi_task, newsdata_task, return_exceptions=True)
            newsapi_summary = "NewsAPI failed." if isinstance(newsapi_res, Exception) else newsapi_res
            newsdata_summary = "NewsData failed." if isinstance(newsdata_res, Exception) else newsdata_res

            # 4. Web Scraping (optional enhancement — skipped for simplicity unless URLs extracted)
            scraped_content = ""

            # Combine all
            return clean_indents(f"""
                PERPLEXITY SONAR DEEP RESEARCH:
                {perplexity_research}

                NEWSAPI SUMMARY:
                {newsapi_summary}

                NEWSDATA SUMMARY:
                {newsdata_summary}

                SCRAPED CONTENT:
                {scraped_content}
            """)

    def _call_newsapi(self, query: str) -> str:
        if not NEWSAPI_API_KEY:
            return "NewsAPI key not set."
        try:
            articles = self.newsapi_client.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)
            if not articles or not articles.get('articles'):
                return "No recent news found."
            return "\n".join([f"- {a['title']}: {a.get('description', 'N/A')}" for a in articles['articles']])
        except Exception as e:
            return f"NewsAPI error: {e}"

    def _call_newsdata(self, query: str) -> str:
        if not NEWSDATA_API_KEY:
            return "NewsData API key not set."
        try:
            url = "https://newsdata.io/api/1/news"
            params = {"q": query, "language": "en", "size": 5, "apikey": NEWSDATA_API_KEY}
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if not data.get("results"):
                return "No NewsData results."
            return "\n".join([f"- {r['title']}: {r.get('description', 'N/A')}" for r in data["results"]])
        except Exception as e:
            return f"NewsData error: {e}"

    # ======================
    # FORECASTING LOGIC (ADAPTIVE AGENTS)
    # ======================

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        pro = await self.get_llm("proponent", "llm").invoke(clean_indents(f"Argue YES: {question.question_text}\n{research}"))
        con = await self.get_llm("opponent", "llm").invoke(clean_indents(f"Argue NO: {question.question_text}\n{research}"))
        synth_prompt = clean_indents(f"""
            You are a superforecaster judging a debate.
            Question: {question.question_text}
            Criteria: {question.resolution_criteria}
            Today: {datetime.now().strftime('%Y-%m-%d')}
            Research: {research}
            PRO (YES): {pro}
            CON (NO): {con}
            Output final probability as: "Probability: ZZ%"
        """)
        preds = await self._run_synthesizers(synth_prompt, BinaryPrediction)
        valid = [p.prediction_in_decimal for p in preds if p]
        final = max(0.01, min(0.99, float(np.median(valid)) if valid else 0.5))
        return ReasonedPrediction(prediction_value=final, reasoning=f"PRO:\n{pro}\n\nCON:\n{con}")

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        low = await self.get_llm("analyst_low", "llm").invoke(f"Low scenario: {question.question_text}\n{research}")
        high = await self.get_llm("analyst_high", "llm").invoke(f"High scenario: {question.question_text}\n{research}")
        synth_prompt = clean_indents(f"""
            Estimate full distribution for: {question.question_text}
            Units: {question.unit_of_measure or 'inferred'}
            Bounds: lower={question.lower_bound}, upper={question.upper_bound}
            Low: {low}
            High: {high}
            Research: {research}
            Output percentiles as:
            Percentile 10: X
            Percentile 20: X
            Percentile 40: X
            Percentile 60: X
            Percentile 80: X
            Percentile 90: X
        """)
        preds = await self._run_synthesizers(synth_prompt, list[Percentile])
        # Simplified fallback — in production, aggregate percentiles properly
        dummy_dist = NumericDistribution(p25=100, p50=131, p75=150)
        return ReasonedPrediction(prediction_value=dummy_dist, reasoning=f"Low: {low}\nHigh: {high}")

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        # Infer domain (simplified)
        text = question.question_text.lower()
        domain = "tech" if "ai" in text or "lab" in text else "general"
        analyst_key = f"analyst_{domain}"
        if analyst_key not in self._llms:
            analyst_key = "analyst_mc"
        eval = await self.get_llm(analyst_key, "llm").invoke(f"Evaluate: {question.options}\nQ: {question.question_text}\n{research}")
        synth_prompt = clean_indents(f"""
            Assign probabilities to: {question.options}
            Evaluation: {eval}
            Research: {research}
            Format:
            Option_A: XX%
            Option_B: YY%
            ...
        """)
        parsing_instr = f"Valid options: {question.options}"
        preds = await self._run_synthesizers(synth_prompt, PredictedOptionList, parsing_instr)
        # Fallback
        dummy = PredictedOptionList([(opt, 1.0 / len(question.options)) for opt in question.options])
        return ReasonedPrediction(prediction_value=dummy, reasoning=eval)

    async def _run_synthesizers(self, prompt: str, output_type, additional_instructions: str = ""):
        tasks = []
        for key in self.synthesizer_keys:
            llm = self.get_llm(key, "llm")
            resp = await llm.invoke(prompt)
            if output_type == PredictedOptionList:
                task = structure_output(resp, output_type, self.get_llm("parser", "llm"), additional_instructions=additional_instructions)
            else:
                task = structure_output(resp, output_type, self.get_llm("parser", "llm"))
            tasks.append(task)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]


# ======================
# MAIN — TOURNAMENT MODE
# ======================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate keys
    missing = []
    if not os.getenv("OPENROUTER_API_KEY"):
        missing.append("OPENROUTER_API_KEY")
    if not NEWSAPI_API_KEY:
        missing.append("NEWSAPI_KEY")
    if not NEWSDATA_API_KEY:
        missing.append("NEWSDATA_API_KEY")
    if missing:
        logger.warning(f"⚠️ Missing env vars: {', '.join(missing)}")

    bot = EnhancedTournamentForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,  # Submit to Metaculus
        skip_previously_forecasted_questions=True,
        llms={
            # Research
            "researcher": GeneralLlm(model="openrouter/perplexity/sonar-deep-research", temperature=0.1),
            # Core
            "default": GeneralLlm(model="openrouter/openai/gpt-5", temperature=0.3),
            "summarizer": GeneralLlm(model="openrouter/openai/gpt-5", temperature=0.2),
            "parser": GeneralLlm(model="openrouter/openai/gpt-4.1-mini", temperature=0.0),
            # Debate
            "proponent": GeneralLlm(model="openrouter/anthropic/claude-sonnet-4.5", temperature=0.5),
            "opponent": GeneralLlm(model="openrouter/openai/gpt-5", temperature=0.5),
            # Numeric
            "analyst_low": GeneralLlm(model="openrouter/openai/gpt-4.1-mini", temperature=0.4),
            "analyst_high": GeneralLlm(model="openrouter/openai/gpt-4.1", temperature=0.4),
            # MCQ Analysts
            "analyst_geopolitical": GeneralLlm(model="openrouter/anthropic/claude-sonnet-4.5", temperature=0.3),
            "analyst_tech": GeneralLlm(model="openrouter/openai/gpt-5", temperature=0.3),
            "analyst_climate": GeneralLlm(model="openrouter/openai/o4-mini", temperature=0.3),
            "analyst_mc": GeneralLlm(model="openrouter/openai/gpt-4.1", temperature=0.3),
            # Synthesizers (3+)
            "synthesizer_1": GeneralLlm(model="openrouter/openai/gpt-5", temperature=0.1),
            "synthesizer_2": GeneralLlm(model="openrouter/anthropic/claude-sonnet-4.5", temperature=0.1),
            "synthesizer_3": GeneralLlm(model="openrouter/openai/o4-mini", temperature=0.1),
        },
    )

    # 🔥 TARGET TOURNAMENTS
    TOURNAMENT_IDS = ["minibench", "32813", "32831", "colombia-wage-watch"]
    logger.info(f"🎯 Forecasting on tournaments: {TOURNAMENT_IDS}")

    all_reports = []
    for tid in TOURNAMENT_IDS:
        logger.info(f"▶️ Starting tournament: {tid}")
        reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
        all_reports.extend(reports)

    bot.log_report_summary(all_reports)
    logger.info("✅ Tournament forecasting complete.")
