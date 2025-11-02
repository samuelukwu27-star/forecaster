import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

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
NEWSAPI_API_KEY = os.getenv("NEWSAPI_KEY")


class FinalTournamentBot2025(ForecastBot):
    """
    Final bot for Fall 2025 tournaments.
    Includes GPT-5 and Claude-4.5 as requested (placeholders).
    Uses real Perplexity for live research.
    Forecasts on: minibench, 32813, 32831, colombia-wage-watch
    Targets:
      - Q578 / Q40159: 30–35% chance of ≥10% pop drop by 2100
      - Q14333: 131 years oldest human by 2100
      - Q22427: 50.8% for "0 or 1" new AI labs by 2030
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        defaults = super()._llm_config_defaults()
        # BOSS MODE: Include gpt-5 and claude-4.5 as requested
        defaults.update({
            "researcher": "openrouter/perplexity/llama-3.1-sonar-large-128k-online",  # ✅ Real
            "default": "openrouter/openai/gpt-5",                                      # 🔜 Placeholder
            "parser": "openrouter/openai/gpt-4o-mini",                                # ✅ Real

            "proponent": "openrouter/anthropic/claude-4.5-sonnet",                   # 🔜 Placeholder
            "opponent": "openrouter/openai/gpt-5",                                   # 🔜 Placeholder

            "analyst_low": "openrouter/openai/gpt-4o-mini",                          # ✅ Real
            "analyst_high": "openrouter/openai/gpt-5",                               # 🔜 Placeholder

            "analyst_geopolitical": "openrouter/anthropic/claude-4.5-sonnet",       # 🔜 Placeholder
            "analyst_tech": "openrouter/openai/gpt-5",                              # 🔜 Placeholder
            "analyst_climate": "openrouter/openai/gpt-4o-mini",                     # ✅ Real
            "analyst_mc": "openrouter/openai/gpt-5",                                # 🔜 Placeholder

            "synthesizer_1": "openrouter/openai/gpt-5",                             # 🔜 Placeholder
            "synthesizer_2": "openrouter/anthropic/claude-4.5-sonnet",             # 🔜 Placeholder
            "synthesizer_3": "openrouter/openai/gpt-4o",                            # ✅ Real fallback
        })
        return defaults

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY)
        self.synthesizer_keys = [k for k in self._llms.keys() if k.startswith("synthesizer")]
        logger.info(f"Intialized with synthesizers: {self.synthesizer_keys}")

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            researcher_llm = self.get_llm("researcher", "llm")
            prompt = clean_indents(f"""
                You are an assistant to a superforecaster.
                Question: {question.question_text}
                Resolution Criteria: {question.resolution_criteria}
                Fine Print: {question.fine_print}
                Provide a concise, factual summary with recent data.
            """)
            research = await researcher_llm.invoke(prompt)

            if NEWSAPI_API_KEY:
                try:
                    articles = self.newsapi_client.get_everything(
                        q=question.question_text, language='en', sort_by='relevancy', page_size=3
                    )
                    if articles and articles.get('articles'):
                        news = "\n".join([f"- {a['title']}" for a in articles['articles'][:3]])
                        research += f"\n\nRecent News:\n{news}"
                except Exception as e:
                    logger.warning(f"NewsAPI failed: {e}")
            return research

    # --- FORECASTING METHODS (from FallTemplateBot2025 structure) ---

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(f"""
            You are a professional forecaster.
            Question: {question.question_text}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}
            Research: {research}
            Today: {datetime.now().strftime('%Y-%m-%d')}
            (a) Time until resolution
            (b) Status quo outcome
            (c) Scenario for No
            (d) Scenario for Yes
            Final output: "Probability: ZZ%"
        """)
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        pred: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, pred.prediction_in_decimal))
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(f"""
            You are a professional forecaster.
            Question: {question.question_text}
            Options: {question.options}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}
            Research: {research}
            Today: {datetime.now().strftime('%Y-%m-%d')}
            (a) Time until resolution
            (b) Status quo outcome
            (c) Unexpected scenario
            Final output:
            {chr(10).join([f"{opt}: XX%" for opt in question.options])}
        """)
        parsing_instructions = f"Valid options: {question.options}"
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        pred: PredictedOptionList = await structure_output(
            reasoning, PredictedOptionList, self.get_llm("parser", "llm"), parsing_instructions
        )
        return ReasonedPrediction(prediction_value=pred, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        low_msg, high_msg = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(f"""
            You are a professional forecaster.
            Question: {question.question_text}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}
            Units: {question.unit_of_measure or 'inferred'}
            Research: {research}
            Today: {datetime.now().strftime('%Y-%m-%d')}
            {low_msg}
            {high_msg}
            (a) Time until resolution
            (b) Outcome if nothing changed
            (c) Outcome if trend continues
            (d) Expert/market expectations
            (e) Low-outcome scenario
            (f) High-outcome scenario
            Final output:
            Percentile 10: X
            Percentile 20: X
            Percentile 40: X
            Percentile 60: X
            Percentile 80: X
            Percentile 90: X
        """)
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        dist = NumericDistribution.from_question(percentile_list, question)
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        low = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
        high = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
        low_msg = f"The outcome cannot be lower than {low}." if not question.open_lower_bound else f"The question creator thinks it's unlikely to be below {low}."
        high_msg = f"The outcome cannot be higher than {high}." if not question.open_upper_bound else f"The question creator thinks it's unlikely to be above {high}."
        return low_msg, high_msg


# ======================
# MAIN
# ======================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("❌ OPENROUTER_API_KEY is required")
        exit(1)

    bot = FinalTournamentBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=5,  # ← Median of 5 forecasts
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )

    TOURNAMENT_IDS = ["minibench", "32813", "32831", "colombia-wage-watch"]
    all_reports = []
    for tid in TOURNAMENT_IDS:
        logger.info(f"▶️ Forecasting on tournament: {tid}")
        reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
        all_reports.extend(reports)

    bot.log_report_summary(all_reports)
    logger.info("✅ Run completed.")
