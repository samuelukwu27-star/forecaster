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

# -----------------------------
# Logging & API Keys
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

NEWSAPI_API_KEY = os.getenv("NEWSAPI_KEY")


class EnhancedTournamentForecaster(ForecastBot):
    """
    Full-featured forecaster for Metaculus tournaments.
    Supports all question types with real, working models.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY)
        self.synthesizer_keys = [k for k in self._llms.keys() if k.startswith("synthesizer")]
        if len(self.synthesizer_keys) < 3:
            raise ValueError("At least 3 synthesizer models required.")
        logger.info("✅ EnhancedTournamentForecaster initialized")

    def _llm_config_defaults(self) -> dict[str, str]:
        """Register all roles with REAL, working OpenRouter models."""
        defaults = super()._llm_config_defaults()
        defaults.update({
            "default": "openrouter/openai/gpt-4o",
            "summarizer": "openrouter/openai/gpt-4o",
            "parser": "openrouter/openai/gpt-4o-mini",
            # ✅ CORRECT Perplexity model for research
            "researcher": "openrouter/perplexity/llama-3.1-sonar-large-128k-online",

            "proponent": "openrouter/anthropic/claude-3.5-sonnet",
            "opponent": "openrouter/openai/gpt-4o",

            "analyst_low": "openrouter/openai/gpt-4o-mini",
            "analyst_high": "openrouter/openai/gpt-4o",

            "analyst_geopolitical": "openrouter/anthropic/claude-3.5-sonnet",
            "analyst_tech": "openrouter/openai/gpt-4o",
            "analyst_climate": "openrouter/openai/gpt-4o-mini",
            "analyst_mc": "openrouter/openai/gpt-4o",

            "synthesizer_1": "openrouter/openai/gpt-4o",
            "synthesizer_2": "openrouter/anthropic/claude-3.5-sonnet",
            "synthesizer_3": "openrouter/openai/gpt-4o-mini",
        })
        return defaults

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            # Perplexity research (real model)
            researcher_llm = self.get_llm("researcher", "llm")
            pplx_prompt = clean_indents(f"""
                You are a forecasting research assistant.
                Summarize key facts for: {question.question_text}
                Resolution Criteria: {question.resolution_criteria}
                Today: {datetime.now().strftime('%Y-%m-%d')}
                Be concise, factual, and focus on recent trends or data.
            """)
            pplx_research = await researcher_llm.invoke(pplx_prompt)

            # NewsAPI (optional)
            news_summary = "NewsAPI not used (key not set)."
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

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        pro = await self.get_llm("proponent", "llm").invoke(clean_indents(f"Argue YES: {question.question_text}\n{research}"))
        con = await self.get_llm("opponent", "llm").invoke(clean_indents(f"Argue NO: {question.question_text}\n{research}"))
        synth_prompt = clean_indents(f"""
            Judge this debate.
            Question: {question.question_text}
            Criteria: {question.resolution_criteria}
            Today: {datetime.now().strftime('%Y-%m-%d')}
            Research: {research}
            PRO (YES): {pro}
            CON (NO): {con}
            Final output: "Probability: ZZ%"
        """)
        preds = await self._run_synthesizers(synth_prompt, BinaryPrediction)
        valid = [p.prediction_in_decimal for p in preds if p]
        final = max(0.01, min(0.99, float(np.median(valid)) if valid else 0.5))
        return ReasonedPrediction(prediction_value=final, reasoning=f"PRO:\n{pro}\n\nCON:\n{con}")

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        low = await self.get_llm("analyst_low", "llm").invoke(f"Low scenario: {question.question_text}\n{research}")
        high = await self.get_llm("analyst_high", "llm").invoke(f"High scenario: {question.question_text}\n{research}")
        synth_prompt = clean_indents(f"""
            Estimate percentiles for: {question.question_text}
            Units: {question.unit_of_measure or 'inferred'}
            Bounds: [{question.lower_bound}, {question.upper_bound}]
            Low: {low}
            High: {high}
            Research: {research}
            Output ONLY:
            Percentile 10: X
            Percentile 20: X
            Percentile 40: X
            Percentile 60: X
            Percentile 80: X
            Percentile 90: X
        """)
        preds = await self._run_synthesizers(synth_prompt, list[Percentile])
        valid_preds = [p for p in preds if p and len(p) >= 6]
        if not valid_preds:
            mid = (question.lower_bound + question.upper_bound) / 2
            fallback_percentiles = [
                Percentile(10, max(question.lower_bound, mid * 0.5)),
                Percentile(20, max(question.lower_bound, mid * 0.7)),
                Percentile(40, mid * 0.9),
                Percentile(60, mid * 1.1),
                Percentile(80, min(question.upper_bound, mid * 1.3)),
                Percentile(90, min(question.upper_bound, mid * 1.5)),
            ]
            dist = NumericDistribution.from_question(fallback_percentiles, question)
            return ReasonedPrediction(prediction_value=dist, reasoning="Fallback due to parsing failure.")
        
        all_vals = {10: [], 20: [], 40: [], 60: [], 80: [], 90: []}
        for pred in valid_preds:
            for p in pred:
                if p.percentile in all_vals:
                    all_vals[p.percentile].append(p.value)
        aggregated = []
        for pt in [10, 20, 40, 60, 80, 90]:
            vals = all_vals[pt]
            med = float(np.median(vals)) if vals else (question.lower_bound + question.upper_bound) / 2
            aggregated.append(Percentile(pt, med))
        dist = NumericDistribution.from_question(aggregated, question)
        return ReasonedPrediction(prediction_value=dist, reasoning=f"LOW:\n{low}\n\nHIGH:\n{high}")

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        text = question.question_text.lower()
        domain = "tech" if any(kw in text for kw in ["ai", "lab", "model", "algorithm"]) else "general"
        analyst_key = f"analyst_{domain}"
        if analyst_key not in self._llms:
            analyst_key = "analyst_mc"
        evaluation = await self.get_llm(analyst_key, "llm").invoke(clean_indents(f"""
            Evaluate options for: {question.question_text}
            Options: {question.options}
            Research: {research}
        """))
        synth_prompt = clean_indents(f"""
            Assign probabilities to EXACT options: {question.options}
            Evaluation: {evaluation}
            Research: {research}
            Today: {datetime.now().strftime('%Y-%m-%d')}
            Output ONLY:
            {chr(10).join([f"{opt}: XX%" for opt in question.options])}
        """)
        parsing_instructions = f"Valid options: {question.options}. Remove any extra prefixes."
        preds = await self._run_synthesizers(synth_prompt, PredictedOptionList, parsing_instructions)
        
        # ✅ FIXED: No .as_dict — use dict(p) instead
        valid_preds = []
        for p in preds:
            if p and isinstance(p, PredictedOptionList) and len(p) > 0:
                try:
                    pred_dict = dict(p)  # PredictedOptionList is list of (option, prob)
                    if all(opt in pred_dict for opt in question.options):
                        valid_preds.append(pred_dict)
                except Exception as e:
                    logger.warning(f"MCQ parsing error: {e}")
                    continue

        if not valid_preds:
            uniform = [(opt, 1.0 / len(question.options)) for opt in question.options]
            return ReasonedPrediction(
                prediction_value=PredictedOptionList(uniform),
                reasoning="Fallback: uniform due to parsing failure."
            )

        avg_probs = {}
        for opt in question.options:
            probs = [pred.get(opt, 0) for pred in valid_preds]
            avg_probs[opt] = float(np.mean(probs))
        total = sum(avg_probs.values())
        if total > 0:
            avg_probs = {k: v / total for k, v in avg_probs.items()}
        final_pred = PredictedOptionList(list(avg_probs.items()))
        return ReasonedPrediction(prediction_value=final_pred, reasoning=evaluation)

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
# MAIN
# ======================

if __name__ == "__main__":
    if not NEWSAPI_API_KEY:
        logger.warning("⚠️ NEWSAPI_API_KEY not set — news research disabled")

    bot = EnhancedTournamentForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )

    TOURNAMENT_IDS = ["minibench", "32813", "32831", "colombia-wage-watch"]
    logger.info(f"🎯 Forecasting on tournaments: {TOURNAMENT_IDS}")

    all_reports = []
    for tid in TOURNAMENT_IDS:
        logger.info(f"▶️ Starting tournament: {tid}")
        reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
        all_reports.extend(reports)

    bot.log_report_summary(all_reports)
    logger.info("✅ Run completed successfully.")
