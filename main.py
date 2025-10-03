import argparse
import asyncio
import logging
from datetime import datetime
from typing import Literal, Any

import numpy as np
from forecasting_tools import (
    AskNewsSearcher,
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
    SmartSearcher,
    clean_indents,
    structure_output,
)
from newsapi import NewsApiClient
from tavily import TavilyClient
import os

# -----------------------------
# Environment & API Keys
# -----------------------------
NEWSAPI_API_KEY = os.getenv("NEWSAPI_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

logger = logging.getLogger(__name__)


class AdaptiveCommitteeBot2025(ForecastBot):
    """
    An adaptive forecasting bot that routes questions to specialized agent committees
    based on question type and uncertainty. Uses debate for binary, scenario generation
    for numeric, and domain analysts for MCQs. Final prediction is synthesized by a
    diverse committee of 3+ models to reduce bias.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY)
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.synthesizer_keys = [k for k in self._llms.keys() if k.startswith("synthesizer")]
        if len(self.synthesizer_keys) < 3:
            raise ValueError("At least 3 synthesizer models required (e.g., synthesizer_1, synthesizer_2, synthesizer_3).")
        logger.info(f"Initialized with adaptive committee: {len(self.synthesizer_keys)} synthesizers.")

    # ======================
    # RESEARCH LAYER
    # ======================

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            # Use Tavily + NewsAPI for broad coverage
            loop = asyncio.get_running_loop()
            tasks = {
                "tavily": loop.run_in_executor(None, self._call_tavily, question.question_text),
                "news": loop.run_in_executor(None, self._call_newsapi, question.question_text),
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            tavily_res, news_res = results

            tavily_summary = "Tavily search failed." if isinstance(tavily_res, Exception) else tavily_res
            news_summary = "NewsAPI failed." if isinstance(news_res, Exception) else news_res

            raw_research = f"Tavily:\n{tavily_summary}\n\nRecent News:\n{news_summary}"
            return raw_research

    def _call_tavily(self, query: str) -> str:
        if not TAVILY_API_KEY:
            return "Tavily API key not set."
        try:
            res = self.tavily_client.search(query=query, search_depth="advanced", max_results=5)
            return "\n".join([f"- {r['content']}" for r in res.get("results", [])])
        except Exception as e:
            logger.error(f"Tavily error: {e}")
            return f"Tavily error: {e}"

    def _call_newsapi(self, query: str) -> str:
        if not NEWSAPI_API_KEY:
            return "NewsAPI key not set."
        try:
            articles = self.newsapi_client.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)
            if not articles or not articles.get('articles'):
                return "No recent news found."
            return "\n".join([f"- {a['title']}: {a.get('description', 'N/A')}" for a in articles['articles']])
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return f"NewsAPI error: {e}"

    # ======================
    # ADAPTIVE FORECASTING
    # ======================

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        logger.info(f"Launching debate for binary question: {question.page_url}")

        # Proponent & Opponent
        pro_prompt = clean_indents(f"""
            Act as a PROPONENT arguing the outcome will be YES.
            Question: {question.question_text}
            Resolution Criteria: {question.resolution_criteria}
            Research: {research}
            Build the strongest evidence-based case for YES.
        """)
        con_prompt = clean_indents(f"""
            Act as an OPPONENT arguing the outcome will be NO.
            Question: {question.question_text}
            Resolution Criteria: {question.resolution_criteria}
            Research: {research}
            Build the strongest evidence-based case for NO.
        """)

        proponent = await self.get_llm("proponent", "llm").invoke(pro_prompt)
        opponent = await self.get_llm("opponent", "llm").invoke(con_prompt)

        # Synthesis prompt
        synth_prompt = clean_indents(f"""
            You are a superforecaster judging a debate.
            Question: "{question.question_text}"
            Criteria: {question.resolution_criteria}
            Today: {datetime.now().strftime('%Y-%m-%d')}
            Research: {research}
            --- PRO (YES) ---
            {proponent}
            --- CON (NO) ---
            {opponent}
            1. Summarize key evidence from both sides.
            2. Assess which scenario is more plausible given status quo bias.
            3. Output final probability as: "Probability: ZZ%"
        """)

        predictions = await self._run_synthesizers(synth_prompt, BinaryPrediction)
        valid_preds = [p.prediction_in_decimal for p in predictions if p]
        if not valid_preds:
            raise ValueError("All synthesizers failed to produce valid binary prediction.")

        median_pred = float(np.median(valid_preds))
        final_pred = max(0.01, min(0.99, median_pred))
        reasoning = self._format_debate_comment(proponent, opponent)

        return ReasonedPrediction(prediction_value=final_pred, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        logger.info(f"Launching scenario analysis for numeric question: {question.page_url}")

        # Monte Carlo-inspired low/high scenarios
        low_prompt = clean_indents(f"""
            Generate a plausible LOW-END scenario for: {question.question_text}
            Use research to justify why the outcome could be near the lower bound.
            Research: {research}
        """)
        high_prompt = clean_indents(f"""
            Generate a plausible HIGH-END scenario for: {question.question_text}
            Use research to justify why the outcome could be near the upper bound.
            Research: {research}
        """)

        low_scenario = await self.get_llm("analyst_low", "llm").invoke(low_prompt)
        high_scenario = await self.get_llm("analyst_high", "llm").invoke(high_prompt)

        synth_prompt = clean_indents(f"""
            You are a superforecaster estimating a full distribution.
            Question: "{question.question_text}"
            Units: {question.unit_of_measure or 'inferred'}
            Bounds: {self._format_bounds(question)}
            Research: {research}
            Low Scenario: {low_scenario}
            High Scenario: {high_scenario}
            Today: {datetime.now().strftime('%Y-%m-%d')}
            Output percentiles as:
            Percentile 10: X
            Percentile 20: X
            Percentile 40: X
            Percentile 60: X
            Percentile 80: X
            Percentile 90: X
        """)

        predictions = await self._run_synthesizers(synth_prompt, list[Percentile])
        valid_preds = [p for p in predictions if p]
        if not valid_preds:
            raise ValueError("All synthesizers failed numeric parsing.")

        # Aggregate percentiles
        all_p10 = [next((x.value for x in p if x.percentile == 10), None) for p in valid_preds]
        all_p20 = [next((x.value for x in p if x.percentile == 20), None) for p in valid_preds]
        all_p40 = [next((x.value for x in p if x.percentile == 40), None) for p in valid_preds]
        all_p60 = [next((x.value for x in p if x.percentile == 60), None) for p in valid_preds]
        all_p80 = [next((x.value for x in p if x.percentile == 80), None) for p in valid_preds]
        all_p90 = [next((x.value for x in p if x.percentile == 90), None) for p in valid_preds]

        def safe_median(vals):
            clean = [v for v in vals if v is not None]
            return float(np.median(clean)) if clean else 0.0

        percentile_list = [
            Percentile(percentile=10, value=safe_median(all_p10)),
            Percentile(percentile=20, value=safe_median(all_p20)),
            Percentile(percentile=40, value=safe_median(all_p40)),
            Percentile(percentile=60, value=safe_median(all_p60)),
            Percentile(percentile=80, value=safe_median(all_p80)),
            Percentile(percentile=90, value=safe_median(all_p90)),
        ]

        dist = NumericDistribution.from_question(percentile_list, question)
        reasoning = self._format_scenario_comment(low_scenario, high_scenario)
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        logger.info(f"Launching domain analysis for MCQ: {question.page_url}")

        # Route to domain-specialized analyst
        domain = self._infer_domain(question.question_text)
        analyst_key = f"analyst_{domain}"
        if analyst_key not in self._llms:
            analyst_key = "analyst_mc"  # fallback

        eval_prompt = clean_indents(f"""
            You are a {domain.upper()} expert evaluating options for:
            Question: {question.question_text}
            Options: {question.options}
            Research: {research}
            Provide a balanced assessment of each option's likelihood.
        """)

        evaluation = await self.get_llm(analyst_key, "llm").invoke(eval_prompt)

        synth_prompt = clean_indents(f"""
            You are a superforecaster assigning probabilities to options.
            Question: "{question.question_text}"
            Options: {question.options}
            Expert Evaluation: {evaluation}
            Research: {research}
            Assign probabilities that sum to 100%. Format as:
            Option_A: XX%
            Option_B: YY%
            ...
        """)

        parsing_instructions = f"Valid options: {question.options}"
        predictions = await self._run_synthesizers(
            synth_prompt, PredictedOptionList, additional_instructions=parsing_instructions
        )
        valid_preds = [p.as_dict for p in predictions if p]
        if not valid_preds:
            raise ValueError("All MCQ synthesizers failed.")

        # Average probabilities
        avg_probs = {}
        for opt in question.options:
            vals = [pred.get(opt, 0) for pred in valid_preds]
            avg_probs[opt] = float(np.mean(vals))
        total = sum(avg_probs.values())
        if total > 0:
            avg_probs = {k: v / total for k, v in avg_probs.items()}

        final_pred = PredictedOptionList(list(avg_probs.items()))
        reasoning = f"Domain: {domain.upper()}\nExpert Evaluation:\n{evaluation}"
        return ReasonedPrediction(prediction_value=final_pred, reasoning=reasoning)

    def _infer_domain(self, text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["geopolitic", "war", "election", "country", "sanction"]):
            return "geopolitical"
        elif any(kw in text_lower for kw in ["ai", "model", "lab", "algorithm", "compute"]):
            return "tech"
        elif any(kw in text_lower for kw in ["climate", "temperature", "co2", "emission"]):
            return "climate"
        else:
            return "general"

    def _format_bounds(self, q: NumericQuestion) -> str:
        low = q.nominal_lower_bound if q.nominal_lower_bound is not None else q.lower_bound
        high = q.nominal_upper_bound if q.nominal_upper_bound is not None else q.upper_bound
        return f"Lower: {low} (open: {q.open_lower_bound}), Upper: {high} (open: {q.open_upper_bound})"

    async def _run_synthesizers(self, prompt: str, output_type, additional_instructions: str = ""):
        tasks = []
        for key in self.synthesizer_keys:
            llm = self.get_llm(key, "llm")
            response = await llm.invoke(prompt)
            if output_type == PredictedOptionList:
                task = structure_output(response, output_type, self.get_llm("parser", "llm"), additional_instructions=additional_instructions)
            else:
                task = structure_output(response, output_type, self.get_llm("parser", "llm"))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    def _format_debate_comment(self, pro: str, con: str) -> str:
        return f"--- PROPONENT ---\n{pro}\n\n--- OPPONENT ---\n{con}"

    def _format_scenario_comment(self, low: str, high: str) -> str:
        return f"--- LOW SCENARIO ---\n{low}\n\n--- HIGH SCENARIO ---\n{high}"


# ======================
# MAIN EXECUTION
# ======================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run AdaptiveCommitteeBot2025")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "test_questions"],
        default="test_questions",
    )
    args = parser.parse_args()

    bot = AdaptiveCommitteeBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=False,
        skip_previously_forecasted_questions=False,
        llms={
            # Core roles
            "default": GeneralLlm(model="openrouter/openai/gpt-4o", temperature=0.3),
            "parser": GeneralLlm(model="openrouter/openai/gpt-4o-mini", temperature=0.0),
            # Debate agents
            "proponent": GeneralLlm(model="openrouter/anthropic/claude-4.5-sonnet", temperature=0.5),
            "opponent": GeneralLlm(model="openrouter/openai/gpt-5", temperature=0.5),
            # Numeric analysts
            "analyst_low": GeneralLlm(model="openrouter/meta/llama-3-70b", temperature=0.4),
            "analyst_high": GeneralLlm(model="openrouter/openai/gpt-o3", temperature=0.4),
            # MCQ domain analysts
            "analyst_geopolitical": GeneralLlm(model="openrouter/anthropic/claude-4.5-sonnet", temperature=0.3),
            "analyst_tech": GeneralLlm(model="openrouter/openai/gpt-5", temperature=0.3),
            "analyst_climate": GeneralLlm(model="openrouter/meta/llama-3-70b", temperature=0.3),
            "analyst_mc": GeneralLlm(model="openrouter/openai/gpt-4o", temperature=0.3),
            # Synthesizers (diverse ensemble)
            "synthesizer_1": GeneralLlm(model="openrouter/openai/gpt-5", temperature=0.1),
            "synthesizer_2": GeneralLlm(model="openrouter/anthropic/claude-4.5-sonnet", temperature=0.1),
            "synthesizer_3": GeneralLlm(model="openrouter/openai/gpt-o3", temperature=0.1),
        },
    )

    if args.mode == "test_questions":
        TEST_URLS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
        ]
        questions = [MetaculusApi.get_question_by_url(url) for url in TEST_URLS]
        reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))
        bot.log_report_summary(reports)
