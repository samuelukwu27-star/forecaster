import argparse
import asyncio
import logging
import os
import textwrap
import re
import math
from datetime import datetime
from typing import List, Optional, Union, Dict, Any

# AskNews integration (preferred official SDK; fallback to requests if unavailable)
try:
    from asknews_sdk import AskNewsSDK
    ASKNEWS_SDK_AVAILABLE = True
except ImportError:
    ASKNEWS_SDK_AVAILABLE = False
    import requests

# Forecasting tools
from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    PredictedOption,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

# -----------------------------
# Helper: Type-safe median
# -----------------------------
def median(lst: List[Union[float, int]]) -> float:
    numeric_vals = [x for x in lst if isinstance(x, (int, float)) and not isinstance(x, bool)]
    if not numeric_vals:
        raise ValueError("median() arg contains no numeric values")

    sorted_vals = sorted(numeric_vals)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return float(sorted_vals[mid])


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def stdev(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ci90(xs: List[float]) -> tuple[float, float]:
    # Simple normal-approx CI as an uncertainty proxy for ensemble agreement
    m = mean(xs)
    s = stdev(xs)
    se = s / math.sqrt(len(xs)) if xs else 0.0
    z = 1.645
    lo = max(0.0, m - z * se)
    hi = min(1.0, m + z * se)
    return lo, hi


def entropy(probs: Dict[str, float]) -> float:
    # Natural log entropy; lower => sharper distribution
    e = 0.0
    for p in probs.values():
        if p > 0:
            e -= p * math.log(p)
    return e


# -----------------------------
# ASKNEWS QUERY BUILDER
# -----------------------------
def build_asknews_query(question: MetaculusQuestion, max_chars: int = 397) -> str:
    q = (question.question_text or "").strip()
    bg = (question.background_info or "").strip()

    # Remove URLs and extra whitespace
    q = re.sub(r"http\S+", "", q)
    bg = re.sub(r"http\S+", "", bg)
    q = re.sub(r"\s+", " ", q).strip()
    bg = re.sub(r"\s+", " ", bg).strip()

    if len(q) <= max_chars:
        if not bg:
            return q
        candidate = f"{q} — {bg}"
        if len(candidate) <= max_chars:
            return candidate

        # Try to fit a shortened background
        space_for_bg = max_chars - len(q) - 3
        if space_for_bg > 10:
            bg_part = textwrap.shorten(bg, width=space_for_bg, placeholder="…")
            return f"{q} — {bg_part}"
        return q

    # If question itself is too long, take the first sentence
    first_sent = q.split(".")[0].strip()
    if len(first_sent) > max_chars:
        return textwrap.shorten(first_sent, width=max_chars, placeholder="…")

    remaining = max_chars - len(first_sent) - 3
    if remaining > 10 and bg:
        bg_part = textwrap.shorten(bg, width=remaining, placeholder="…")
        combo = f"{first_sent} — {bg_part}"
        if len(combo) <= max_chars:
            return combo

    return textwrap.shorten(q, width=max_chars, placeholder="…")


# -----------------------------
# Logging & AskNews Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("FinalTournamentBot2025")

ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_CLIENT_SECRET = os.getenv("ASKNEWS_CLIENT_SECRET")

ASKNEWS_ENABLED = bool(ASKNEWS_CLIENT_ID and ASKNEWS_CLIENT_SECRET)
if not ASKNEWS_ENABLED:
    logger.warning("ASKNEWS_CLIENT_ID/ASKNEWS_CLIENT_SECRET not set — continuing in LLM-only research mode.")


class FinalTournamentBot2025(ForecastBot):
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    # Only two models: GPT-5.2 and Claude Sonnet 4.5 (via OpenRouter)
    def _llm_config_defaults(self) -> Dict[str, str]:
        defaults = super()._llm_config_defaults()
        defaults.update({
            # Research
            "researcher_gpt": "openrouter/openai/gpt-5.2",
            "researcher_claude": "openrouter/anthropic/claude-sonnet-4.5",

            # Forecasting
            "forecaster_gpt": "openrouter/openai/gpt-5.2",
            "forecaster_claude": "openrouter/anthropic/claude-sonnet-4.5",

            # Parsing
            "parser": "openrouter/anthropic/claude-sonnet-4.5",
        })
        return defaults

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._asknews_client = None
        logger.info("Initialized FinalTournamentBot2025 (AskNews + 2-model ensemble)")

        if ASKNEWS_SDK_AVAILABLE:
            try:
                import asknews_sdk
                logger.info(f"AskNews SDK version: {asknews_sdk.__version__}")
            except Exception:
                pass

        # 5-run schedule for diversity using ONLY the 2 models
        self._run_schedule = ["gpt", "claude", "gpt", "claude", "gpt"]

    def _get_asknews_client(self):
        if self._asknews_client is not None:
            return self._asknews_client

        if not ASKNEWS_ENABLED:
            self._asknews_client = None
            return None

        if ASKNEWS_SDK_AVAILABLE:
            self._asknews_client = AskNewsSDK(
                client_id=ASKNEWS_CLIENT_ID,
                client_secret=ASKNEWS_CLIENT_SECRET,
                scopes=["news"],
            )
        else:
            auth_url = "https://api.asknews.app/v1/oauth/token"
            data = {
                "grant_type": "client_credentials",
                "client_id": ASKNEWS_CLIENT_ID,
                "client_secret": ASKNEWS_CLIENT_SECRET,
                "scope": "news",
            }
            try:
                resp = requests.post(auth_url, data=data, timeout=10)
                resp.raise_for_status()
                token = resp.json()["access_token"]
                self._asknews_client = {"token": token}
            except Exception as e:
                logger.error(f"Failed to authenticate with AskNews API: {e}")
                self._asknews_client = None

        return self._asknews_client

    def _get_research_llm(self, model_tag: str):
        if model_tag == "claude":
            return self.get_llm("researcher_claude", "llm")
        return self.get_llm("researcher_gpt", "llm")

    def _get_forecaster_llm(self, model_tag: str):
        if model_tag == "claude":
            return self.get_llm("forecaster_claude", "llm")
        return self.get_llm("forecaster_gpt", "llm")

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            today_str = datetime.now().strftime("%Y-%m-%d")
            query = build_asknews_query(question)
            logger.debug(f"AskNews query ({len(query)} chars): {repr(query)}")

            # AskNews research
            asknews_summary = "[AskNews disabled]"
            try:
                if ASKNEWS_ENABLED:
                    loop = asyncio.get_running_loop()
                    stories = await loop.run_in_executor(None, self._sync_asknews_search, query)
                    if not stories:
                        asknews_summary = "[AskNews: No recent stories found]"
                    else:
                        snippets = []
                        for i, story in enumerate(stories[:5]):
                            if isinstance(story, dict):
                                title = story.get("title", "Untitled")
                                text = (story.get("text") or "")[:400]
                            else:
                                title = getattr(story, "title", "Untitled")
                                text = (getattr(story, "text", "") or "")[:400]

                            snippet = f"[{i+1}] {title}: {textwrap.shorten(text, width=220, placeholder='…')}"
                            snippets.append(snippet)
                        asknews_summary = "\n".join(snippets)
                        logger.info(f"AskNews succeeded with {len(stories)} stories")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"AskNews research failed: {error_msg}")
                asknews_summary = f"[AskNews error: {error_msg}]"

            # LLM research: use GPT by default for research, with Claude as a secondary check
            research_prompt = clean_indents(f"""
                You are an assistant to a superforecaster. Be concise and factual.
                Question: {question.question_text}
                Resolution Criteria: {question.resolution_criteria}
                Fine Print: {question.fine_print}

                Task:
                - Extract key facts (with dates if present)
                - Identify base rates / reference class if applicable
                - Identify the most decision-relevant variables
                - Note what would falsify the main hypothesis
                - Keep to <= 220 words
            """)

            try:
                llm_main = self._get_research_llm("gpt")
                llm_research_main = await llm_main.invoke(research_prompt)
            except Exception as e:
                llm_research_main = f"[LLM research (GPT) failed: {str(e)}]"

            try:
                llm_check = self._get_research_llm("claude")
                llm_research_check = await llm_check.invoke(research_prompt)
            except Exception as e:
                llm_research_check = f"[LLM research (Claude) failed: {str(e)}]"

            return (
                f"--- ASKNEWS LIVE NEWS (as of {today_str}) ---\n{asknews_summary}\n\n"
                f"--- LLM RESEARCH (GPT-5.2) ---\n{llm_research_main}\n\n"
                f"--- LLM RESEARCH (SONNET-4.5 CHECK) ---\n{llm_research_check}"
            )

    def _sync_asknews_search(self, query: str) -> List[Any]:
        client = self._get_asknews_client()
        if client is None:
            return []

        if ASKNEWS_SDK_AVAILABLE:
            try:
                if hasattr(client.news, "search_news"):
                    response = client.news.search_news(
                        query=query,
                        n_articles=5,
                        return_type="news",
                        method="kw",
                        return_story_text=True,
                    )
                elif hasattr(client.news, "search_stories"):
                    response = client.news.search_stories(
                        query=query,
                        n_articles=5,
                        return_type="news",
                        use_neural_search=False,
                        return_story_text=True,
                        return_story_summary=False,
                    )
                else:
                    raise AttributeError("AskNews client has neither search_news nor search_stories")

                if hasattr(response, "news"):
                    return response.news
                if hasattr(response, "as_dict"):
                    return response.as_dict().get("news", [])

                data = getattr(response, "data", response if isinstance(response, dict) else {})
                return data.get("news", []) if isinstance(data, dict) else []
            except Exception as e:
                logger.error(f"AskNews SDK search error: {e}")
                return []
        else:
            headers = {"Authorization": f"Bearer {client['token']}"}
            params = {
                "q": query,
                "n_articles": 5,
                "sort": "relevance",
                "return_type": "news",
                "use_neural_search": "false",
                "return_story_text": "true",
            }
            try:
                resp = requests.get(
                    "https://api.asknews.app/v1/news",
                    headers=headers,
                    params=params,
                    timeout=15,
                )
                resp.raise_for_status()
                json_resp = resp.json()
                data = json_resp.get("data", json_resp)
                return data.get("news", [])
            except Exception as e:
                logger.error(f"AskNews HTTP search error: {e}")
                return []

    # -----------------------------
    # Forecasting methods (2-model ensemble + stats)
    # -----------------------------
    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str, run_id: int, model_tag: str) -> ReasonedPrediction[float]:
        prompt = clean_indents(f"""
            You are a professional superforecaster. Be decisive and avoid hedging.
            You must output a single final probability and it must be internally consistent with your reasoning.

            Run: {run_id} / Model: {model_tag}
            Requirement: Take a distinct perspective from other runs (different reference class or causal model).

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}

            Research:
            {research}

            Today: {datetime.now().strftime('%Y-%m-%d')}

            Provide:
            (a) Time until resolution
            (b) Base rate / reference class
            (c) Key drivers and current status quo
            (d) Best case for NO
            (e) Best case for YES

            Final output EXACTLY:
            Probability: ZZ%
        """)
        llm = self._get_forecaster_llm(model_tag)
        reasoning = await llm.invoke(prompt)

        pred: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )

        try:
            val = float(pred.prediction_in_decimal)
        except (ValueError, TypeError):
            val = 0.5

        decimal_pred = max(0.01, min(0.99, val))
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str, run_id: int, model_tag: str) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(f"""
            You are a professional superforecaster. Be decisive and avoid hedging.
            You must allocate probabilities across options that sum to 100%.

            Run: {run_id} / Model: {model_tag}
            Requirement: Take a distinct perspective from other runs (different reference class or causal model).

            Question: {question.question_text}
            Options: {question.options}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}

            Research:
            {research}

            Today: {datetime.now().strftime('%Y-%m-%d')}

            Provide:
            (a) Time until resolution
            (b) Status quo and base rates
            (c) Surprise scenario

            Final output EXACTLY as:
            {chr(10).join([f"{opt}: XX%" for opt in question.options])}
        """)
        parsing_instructions = f"Valid options: {question.options}"
        llm = self._get_forecaster_llm(model_tag)
        reasoning = await llm.invoke(prompt)

        pred: PredictedOptionList = await structure_output(
            reasoning, PredictedOptionList, self.get_llm("parser", "llm"), parsing_instructions
        )
        return ReasonedPrediction(prediction_value=pred, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str, run_id: int, model_tag: str) -> ReasonedPrediction[NumericDistribution]:
        low_msg, high_msg = self._create_upper_and_lower_bound_messages(question)
        unit = getattr(question, "unit_of_measure", "inferred")

        prompt = clean_indents(f"""
            You are a professional superforecaster. Be decisive and avoid hedging.
            You must output coherent percentiles (monotonic increasing).

            Run: {run_id} / Model: {model_tag}
            Requirement: Take a distinct perspective from other runs (different reference class or causal model).

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}
            Units: {unit}

            Research:
            {research}

            Today: {datetime.now().strftime('%Y-%m-%d')}

            {low_msg}
            {high_msg}

            Provide:
            (a) Time until resolution
            (b) Base rate / reference class
            (c) Trend continuation case
            (d) Expert/market expectations
            (e) Low-outcome scenario
            (f) High-outcome scenario

            Final output EXACTLY:
            Percentile 10: X
            Percentile 20: X
            Percentile 40: X
            Percentile 60: X
            Percentile 80: X
            Percentile 90: X
        """)

        llm = self._get_forecaster_llm(model_tag)
        reasoning = await llm.invoke(prompt)

        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )

        clean_percentiles: list[Percentile] = []
        for p in percentile_list:
            try:
                v_str = str(p.value).replace(",", "").replace("%", "").strip()
                val = float(v_str)
                clean_percentiles.append(Percentile(value=val, percentile=p.percentile))
            except (ValueError, TypeError):
                continue

        if not clean_percentiles:
            mid = 0.0
            if (question.lower_bound is not None) and (question.upper_bound is not None):
                try:
                    mid = (float(question.lower_bound) + float(question.upper_bound)) / 2.0
                except Exception:
                    mid = 0.0
            clean_percentiles = [Percentile(value=mid, percentile=0.5)]

        dist = NumericDistribution.from_question(clean_percentiles, question)
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(self, question: NumericQuestion) -> tuple[str, str]:
        low = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
        high = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
        low_msg = (
            f"The outcome cannot be lower than {low}."
            if not question.open_lower_bound
            else f"The question creator thinks it's unlikely to be below {low}."
        )
        high_msg = (
            f"The outcome cannot be higher than {high}."
            if not question.open_upper_bound
            else f"The question creator thinks it's unlikely to be above {high}."
        )
        return low_msg, high_msg

    # -----------------------------
    # Type-safe aggregation + stats
    # -----------------------------
    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        predictions: list[Any] = []
        reasonings: list[str] = []

        # 5 runs alternating ONLY between GPT-5.2 and Sonnet-4.5
        for i, model_tag in enumerate(self._run_schedule, start=1):
            try:
                if isinstance(question, BinaryQuestion):
                    pred = await self._run_forecast_on_binary(question, research, run_id=i, model_tag=model_tag)
                elif isinstance(question, MultipleChoiceQuestion):
                    pred = await self._run_forecast_on_multiple_choice(question, research, run_id=i, model_tag=model_tag)
                elif isinstance(question, NumericQuestion):
                    pred = await self._run_forecast_on_numeric(question, research, run_id=i, model_tag=model_tag)
                else:
                    raise ValueError(f"Unsupported question type: {type(question)}")

                predictions.append(pred.prediction_value)
                reasonings.append(pred.reasoning)
            except Exception as e:
                logger.error(f"Individual forecaster failed (run {i}, model={model_tag}): {e}")
                continue

        if not predictions:
            raise RuntimeError("All forecasters failed.")

        final_pred: Any = None

        if isinstance(question, BinaryQuestion):
            numeric_preds = [p for p in predictions if isinstance(p, (int, float)) and not isinstance(p, bool)]
            if not numeric_preds:
                numeric_preds = [0.5]

            med = median(numeric_preds)
            m = mean([float(x) for x in numeric_preds])
            s = stdev([float(x) for x in numeric_preds])
            lo, hi = ci90([float(x) for x in numeric_preds])

            # "Very confident" presentation, backed by agreement stats
            stats_line = f"[stats] n={len(numeric_preds)} mean={m:.3f} median={med:.3f} sd={s:.3f} ci90=({lo:.3f},{hi:.3f})"
            final_pred = ReasonedPrediction(
                prediction_value=max(0.01, min(0.99, float(med))),
                reasoning=stats_line + " | " + " | ".join(reasonings),
            )

        elif isinstance(question, MultipleChoiceQuestion):
            options = question.options
            per_option_samples: Dict[str, List[float]] = {opt: [] for opt in options}

            # Collect samples WITHOUT defaulting missing options to 0
            for p in predictions:
                if isinstance(p, PredictedOptionList):
                    pred_dict = {po.option_name.strip(): po.probability for po in p.predicted_options}
                    for opt in options:
                        # normalize name match a bit
                        prob = pred_dict.get(opt) or pred_dict.get(opt.strip())
                        if prob is not None and isinstance(prob, (int, float)) and not isinstance(prob, bool):
                            per_option_samples[opt].append(float(prob))

            # Median per option across available samples; if missing entirely, give tiny floor
            avg_probs: Dict[str, float] = {}
            for opt in options:
                if per_option_samples[opt]:
                    avg_probs[opt] = median(per_option_samples[opt])
                else:
                    avg_probs[opt] = 0.0001

            # Normalize
            total = sum(avg_probs.values())
            if total > 0:
                avg_probs = {k: v / total for k, v in avg_probs.items()}

            ent = entropy(avg_probs)
            stats_line = f"[stats] n_runs={len(predictions)} entropy={ent:.3f} (lower=more confident)"

            predicted_options_list = [
                PredictedOption(option_name=opt, probability=prob)
                for opt, prob in avg_probs.items()
            ]
            final_pred = ReasonedPrediction(
                prediction_value=PredictedOptionList(predicted_options=predicted_options_list),
                reasoning=stats_line + " | " + " | ".join(reasonings),
            )

        elif isinstance(question, NumericQuestion):
            target_pts = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            median_percentiles: list[Percentile] = []

            for pt in target_pts:
                vals: list[float] = []
                for p in predictions:
                    if isinstance(p, NumericDistribution):
                        for item in p.declared_percentiles:
                            try:
                                val = float(str(item.value).replace(",", "").strip())
                            except (ValueError, TypeError):
                                continue

                            try:
                                p_val = float(item.percentile)
                                is_match = abs(p_val - pt) < 0.01 or abs(p_val - (pt * 100.0)) < 1.0
                                if is_match:
                                    vals.append(val)
                            except Exception:
                                continue

                if vals:
                    med_val = median(vals)
                else:
                    l = question.lower_bound if question.lower_bound is not None else 0.0
                    u = question.upper_bound if question.upper_bound is not None else 100.0
                    try:
                        med_val = (float(l) + float(u)) / 2.0
                    except Exception:
                        med_val = 0.0

                median_percentiles.append(Percentile(percentile=pt, value=med_val))

            # Enforce monotonic percentiles
            median_percentiles.sort(key=lambda x: float(x.percentile))
            for i in range(1, len(median_percentiles)):
                if median_percentiles[i].value < median_percentiles[i - 1].value:
                    median_percentiles[i].value = median_percentiles[i - 1].value

            # Spread metric (uncertainty proxy)
            p10 = next((p.value for p in median_percentiles if abs(float(p.percentile) - 0.1) < 1e-9), None)
            p90 = next((p.value for p in median_percentiles if abs(float(p.percentile) - 0.9) < 1e-9), None)
            spread = (p90 - p10) if (p10 is not None and p90 is not None) else float("nan")
            stats_line = f"[stats] n_runs={len(predictions)} p10={p10:.3f} p90={p90:.3f} spread(p90-p10)={spread:.3f}"

            final_dist = NumericDistribution.from_question(median_percentiles, question)
            final_pred = ReasonedPrediction(
                prediction_value=final_dist,
                reasoning=stats_line + " | " + " | ".join(reasonings),
            )

        else:
            final_pred = ReasonedPrediction(prediction_value=predictions[0], reasoning=" | ".join(reasonings))

        return final_pred


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    litellm_logger = logging.getLogger("litellm")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(description="Run FinalTournamentBot2025 (AskNews + 2-model ensemble)")
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        type=str,
        default=["minibench", "32916", "market-pulse-26q1", "ACX2026"],
    )
    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("❌ OPENROUTER_API_KEY is required")
        raise SystemExit(1)

    bot = FinalTournamentBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )

    async def run_all():
        all_reports = []
        for tid in args.tournament_ids:
            logger.info(f"▶️ Forecasting on tournament: {tid}")
            reports = await bot.forecast_on_tournament(tid, return_exceptions=True)
            all_reports.extend(reports)
        return all_reports

    try:
        reports = asyncio.run(run_all())
        bot.log_report_summary(reports)
        logger.info("✅ FinalTournamentBot2025 run completed.")
    except Exception as e:
        logger.error(f"❌ Fatal error during execution: {e}")
        raise SystemExit(1))
