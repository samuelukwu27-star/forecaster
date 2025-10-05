import argparse
import asyncio
import logging
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

logger = logging.getLogger(__name__)


class PerplexityHybridBot2025(ForecastBot):
    """
    Hybrid forecasting bot using Perplexity (via OpenRouter) for live research
    and a 5-model ensemble for robust prediction synthesis.
    
    Researcher: openrouter/thedrummer/cydonia-24b-v4.1
    """

    def _llm_config_defaults(self) -> dict[str, str]:
        defaults = super()._llm_config_defaults()
        defaults.update({
            # Researcher: Perplexity with live web search
            "researcher": "openrouter/thedrummer/cydonia-24b-v4.1",

            # Forecasting pipeline
            "default": "openrouter/openai/gpt-5",
            "parser": "openrouter/openai/gpt-4o-mini",

            "proponent": "openrouter/anthropic/claude-3.5-sonnet",
            "opponent": "openrouter/openai/gpt-4o",

            "analyst_low": "openrouter/anthropic/claude-opus-4.1",
            "analyst_high": "openrouter/openai/gpt-4o",

            "analyst_geopolitical": "openrouter/anthropic/claude-3.5-sonnet",
            "analyst_tech": "openrouter/openai/gpt-5",
            "analyst_climate": "openrouter/openai/gpt-4o-mini",
            "analyst_mc": "openrouter/openai/gpt-5",

            # 5 Synthesizers for aggregation
            "synthesizer_1": "openrouter/openai/gpt-5",
            "synthesizer_2": "openrouter/anthropic/claude-sonnet-4.5",
            "synthesizer_3": "openrouter/openai/gpt-5",
            "synthesizer_4": "openrouter/anthropic/claude-sonnet-4",
            "synthesizer_5": "openrouter/openai/gpt-4o-mini",
        })
        return defaults

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.synthesizer_keys = [k for k in self._llms.keys() if k.startswith("synthesizer")]
        if len(self.synthesizer_keys) < 3:
            logger.warning("Fewer than 3 synthesizers found — may reduce robustness.")
        logger.info(f"Initialized with {len(self.synthesizer_keys)} synthesizers.")

    async def run_research(self, question: MetaculusQuestion) -> str:
        researcher_llm = self.get_llm("researcher", "llm")
        prompt = clean_indents(f"""
            You are a forecasting research assistant.
            Provide a concise, factual summary with recent data for:
            Question: {question.question_text}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}
            Today: {datetime.now().strftime('%Y-%m-%d')}
            Focus on trends, breakthroughs, or events that could affect the outcome.
            If the question is highly speculative or about the distant future with no current data, state that clearly.
        """)
        research = await researcher_llm.invoke(prompt)
        logger.info(f"Research for {question.page_url}:\n{research}")
        return research

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        pro = await self.get_llm("proponent", "llm").invoke(
            clean_indents(f"Argue YES: {question.question_text}\n{research}")
        )
        con = await self.get_llm("opponent", "llm").invoke(
            clean_indents(f"Argue NO: {question.question_text}\n{research}")
        )
        synth_prompt = clean_indents(f"""
            Judge this debate.
            Question: {question.question_text}
            Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}
            Today: {datetime.now().strftime('%Y-%m-%d')}
            Research: {research}
            PRO (YES): {pro}
            CON (NO): {con}
            Final output: "Probability: ZZ%"
        """)
        preds = await self._run_synthesizers(synth_prompt, BinaryPrediction)
        valid = [p.prediction_in_decimal for p in preds if p]
        final = max(0.01, min(0.99, float(np.median(valid)) if valid else 0.5))
        reasoning = f"PRO:\n{pro}\n\nCON:\n{con}"
        return ReasonedPrediction(prediction_value=final, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        low = await self.get_llm("analyst_low", "llm").invoke(
            f"Low scenario: {question.question_text}\n{research}"
        )
        high = await self.get_llm("analyst_high", "llm").invoke(
            f"High scenario: {question.question_text}\n{research}"
        )
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
            # ✅ FIXED: Use keyword arguments for Percentile
            fallback_percentiles = [
                Percentile(percentile=10, value=max(question.lower_bound, mid * 0.5)),
                Percentile(percentile=20, value=max(question.lower_bound, mid * 0.7)),
                Percentile(percentile=40, value=mid * 0.9),
                Percentile(percentile=60, value=mid * 1.1),
                Percentile(percentile=80, value=min(question.upper_bound, mid * 1.3)),
                Percentile(percentile=90, value=min(question.upper_bound, mid * 1.5)),
            ]
            dist = NumericDistribution.from_question(fallback_percentiles, question)
            return ReasonedPrediction(prediction_value=dist, reasoning="Fallback due to parsing failure.")
        
        all_vals = {10: [], 20: [], 40: [], 60: [], 80: [], 90: []}
        for pred in valid_preds:
            for p in pred:
                if hasattr(p, 'percentile') and hasattr(p, 'value') and p.percentile in all_vals:
                    all_vals[p.percentile].append(p.value)
        aggregated = []
        for pt in [10, 20, 40, 60, 80, 90]:
            vals = all_vals[pt]
            med = float(np.median(vals)) if vals else (question.lower_bound + question.upper_bound) / 2
            aggregated.append(Percentile(percentile=pt, value=med))
        dist = NumericDistribution.from_question(aggregated, question)
        reasoning = f"LOW:\n{low}\n\nHIGH:\n{high}"
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

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
        
        valid_preds = []
        for p in preds:
            if p and isinstance(p, PredictedOptionList) and len(p) > 0:
                try:
                    pred_dict = dict(p)
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(description="Run PerplexityHybridBot2025")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Run mode",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    bot = PerplexityHybridBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )

    if run_mode == "tournament":
        seasonal = asyncio.run(
            bot.forecast_on_tournament(MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True)
        )
        minibench = asyncio.run(
            bot.forecast_on_tournament(MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True)
        )
        market_pulse = asyncio.run(
            bot.forecast_on_tournament("market-pulse-25q4", return_exceptions=True)
        )
        reports = seasonal + minibench + market_pulse
    elif run_mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(
            bot.forecast_on_tournament(MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True)
        )
    elif run_mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
        ]
        bot.skip_previously_forecasted_questions = False
        questions = [MetaculusApi.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
        reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))

    bot.log_report_summary(reports)
    logger.info("✅ Perplexity Hybrid run completed.")
