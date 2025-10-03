import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

import numpy as np
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

logger = logging.getLogger(__name__)

# Use only NewsAPI
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")


class EnhancedTournamentForecaster(ForecastBot):
    """
    Real-model, tournament-focused forecaster.
    Research: NewsAPI + Perplexity (real model) via OpenRouter.
    Models: Real OpenRouter IDs that match your intent.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY)
        self.synthesizer_keys = [k for k in self._llms.keys() if k.startswith("synthesizer")]
        if len(self.synthesizer_keys) < 3:
            raise ValueError("Need 3+ synthesizers")
        logger.info("✅ Real-Model Forecaster Ready")

    def _llm_config_defaults(self) -> dict[str, str]:
        """Register all roles with REAL OpenRouter models that match your intent."""
        defaults = super()._llm_config_defaults()
        defaults.update({
            # Map your names to real models
            "default": "openrouter/openai/gpt-5",
            "summarizer": "openrouter/openai/gpt-4o",
            "parser": "openrouter/openai/gpt-4o-mini",
            "researcher": "openrouter/perplexity/llama-3.1-sonar-large-128k-online",  # REAL Perplexity model

            "proponent": "openrouter/anthropic/claude-3.5-sonnet",
            "opponent": "openrouter/openai/gpt-4o",

            "analyst_low": "openrouter/openai/gpt-4o-mini",
            "analyst_high": "openrouter/openai/gpt-5",

            "analyst_geopolitical": "openrouter/anthropic/claude-3.5-sonnet",
            "analyst_tech": "openrouter/openai/gpt-5",
            "analyst_climate": "openrouter/openai/gpt-4o-mini",
            "analyst_mc": "openrouter/openai/gpt-5",

            "synthesizer_1": "openrouter/openai/gpt-5",
            "synthesizer_2": "openrouter/anthropic/claude-3.5-sonnet",
            "synthesizer_3": "openrouter/openai/gpt-4o-mini",
        })
        return defaults

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            # 1. Perplexity (real model via OpenRouter)
            researcher_llm = self.get_llm("researcher", "llm")
            pplx_research = await researcher_llm.invoke(clean_indents(f"""
                You are a forecasting research assistant.
                Summarize key facts for: {question.question_text}
                Criteria: {question.resolution_criteria}
                Today: {datetime.now().strftime('%Y-%m-%d')}
                Be concise, factual, and cite trends or data if known.
            """))

            # 2. NewsAPI
            news_summary = "NewsAPI failed."
            if NEWSAPI_API_KEY:
                try:
                    articles = self.newsapi_client.get_everything(
                        q=question.question_text, language='en', sort_by='relevancy', page_size=5
                    )
                    if articles and articles.get('articles'):
                        news_summary = "\n".join([
                            f"- {a['title']}: {a.get('description', 'N/A')}" 
                            for a in articles['articles']
                        ])
                    else:
                        news_summary = "No recent news found."
                except Exception as e:
                    news_summary = f"NewsAPI error: {e}"

            return clean_indents(f"""
                PERPLEXITY RESEARCH:
                {pplx_research}

                NEWSAPI:
                {news_summary}
            """)

    # ... [forecasting methods same as before — use get_llm("proponent"), etc.] ...

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        pro = await self.get_llm("proponent", "llm").invoke(f"Argue YES: {question.question_text}\n{research}")
        con = await self.get_llm("opponent", "llm").invoke(f"Argue NO: {question.question_text}\n{research}")
        synth_prompt = clean_indents(f"Debate: {question.question_text}\nPRO: {pro}\nCON: {con}\nProbability: ZZ%")
        preds = await self._run_synthesizers(synth_prompt, BinaryPrediction)
        valid = [p.prediction_in_decimal for p in preds if p]
        final = max(0.01, min(0.99, float(np.median(valid)) if valid else 0.5))
        return ReasonedPrediction(prediction_value=final, reasoning=f"PRO:\n{pro}\n\nCON:\n{con}")

    # (Include _run_forecast_on_numeric and _run_forecast_on_multiple_choice similarly)

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
# MAIN — TOURNAMENTS
# ======================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if not NEWSAPI_API_KEY:
        logger.warning("⚠️ NEWSAPI_API_KEY not set")

    bot = EnhancedTournamentForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        # No llms dict needed — defaults cover all roles
    )

    TOURNAMENT_IDS = ["minibench", "32813", "32831", "colombia-wage-watch"]
    logger.info(f"🎯 Forecasting on: {TOURNAMENT_IDS}")

    all_reports = []
    for tid in TOURNAMENT_IDS:
        reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
        all_reports.extend(reports)

    bot.log_report_summary(all_reports)
