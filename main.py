import argparse
import asyncio
import logging
import os
import textwrap
import re
from datetime import datetime
from typing import List, Literal, Optional, Union, Dict, Any

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
    GeneralLlm,
    MetaculusApi,
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
    # ✅ Filter out non-numeric (e.g., fallback strings, None, bools)
    numeric_vals = [
        x for x in lst 
        if isinstance(x, (int, float)) and not isinstance(x, bool)
    ]
    if not numeric_vals:
        # Return a safe fallback (e.g., 0.5) or raise strict error depending on pref
        # Raising error here is handled by the caller's try/except
        raise ValueError("median() arg contains no numeric values")
    
    sorted_vals = sorted(numeric_vals)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    else:
        return float(sorted_vals[mid])


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
        else:
            return q

    # If question itself is too long, take the first sentence
    first_sent = q.split('.')[0].strip()
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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FinalTournamentBot2025")

ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_CLIENT_SECRET = os.getenv("ASKNEWS_CLIENT_SECRET")

if not (ASKNEWS_CLIENT_ID and ASKNEWS_CLIENT_SECRET):
    raise EnvironmentError("ASKNEWS_CLIENT_ID and ASKNEWS_CLIENT_SECRET must be set.")


class FinalTournamentBot2025(ForecastBot):
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> Dict[str, str]:
        defaults = super()._llm_config_defaults()
        defaults.update({
            "researcher": "openrouter/openai/gpt-4o", 
            "default": "openrouter/openai/gpt-4o",
            "parser": "openrouter/openai/gpt-4o-mini",
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
            "synthesizer_3": "openrouter/openai/gpt-4o",
        })
        return defaults

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._asknews_client = None
        logger.info("Initialized FinalTournamentBot2025 (AskNews-powered)")
        if ASKNEWS_SDK_AVAILABLE:
            import asknews_sdk
            logger.info(f"AskNews SDK version: {asknews_sdk.__version__}")

    def _get_asknews_client(self):
        if self._asknews_client is not None:
            return self._asknews_client

        if ASKNEWS_SDK_AVAILABLE:
            self._asknews_client = AskNewsSDK(
                client_id=ASKNEWS_CLIENT_ID,
                client_secret=ASKNEWS_CLIENT_SECRET,
                scopes=["news"]
            )
        else:
            auth_url = "https://api.asknews.app/v1/oauth/token"
            data = {
                "grant_type": "client_credentials",
                "client_id": ASKNEWS_CLIENT_ID,
                "client_secret": ASKNEWS_CLIENT_SECRET,
                "scope": "news"
            }
            try:
                resp = requests.post(auth_url, data=data, timeout=10)
                resp.raise_for_status()
                token = resp.json()["access_token"]
                self._asknews_client = {"token": token}
            except Exception as e:
                logger.error(f"Failed to authenticate with AskNews API: {e}")
                raise

        return self._asknews_client

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            today_str = datetime.now().strftime("%Y-%m-%d")

            query = build_asknews_query(question)
            logger.debug(f"AskNews query ({len(query)} chars): {repr(query)}")

            asknews_summary = "[AskNews research pending]"

            try:
                loop = asyncio.get_running_loop()
                stories = await loop.run_in_executor(
                    None,
                    self._sync_asknews_search,
                    query
                )

                if not stories:
                    asknews_summary = "[AskNews: No recent stories found]"
                else:
                    snippets = []
                    for i, story in enumerate(stories[:5]):
                        # Robust extraction for both Object (SDK) and Dict (Requests)
                        if isinstance(story, dict):
                            title = story.get("title", "Untitled")
                            text = (story.get("text") or "")[:200]
                        else:
                            title = getattr(story, "title", "Untitled")
                            text = getattr(story, "text", "")[:200]
                        
                        snippet = f"[{i+1}] {title}: {textwrap.shorten(text, width=180, placeholder='…')}"
                        snippets.append(snippet)
                    asknews_summary = "\n".join(snippets)
                    logger.info(f"AskNews succeeded with {len(stories)} stories")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"AskNews research failed: {error_msg}")
                asknews_summary = f"[AskNews error: {error_msg}]"

            # LLM research
            researcher_llm = self.get_llm("researcher", "llm")
            prompt = clean_indents(f"""
                You are an assistant to a superforecaster.
                Question: {question.question_text}
                Resolution Criteria: {question.resolution_criteria}
                Fine Print: {question.fine_print}
                Provide a concise, factual summary with recent data.
            """)
            try:
                llm_research = await researcher_llm.invoke(prompt)
            except Exception as e:
                llm_research = f"[LLM research failed: {str(e)}]"

            return (
                f"--- ASKNEWS LIVE NEWS (as of {today_str}) ---\n{asknews_summary}\n\n"
                f"--- LLM RESEARCH SUMMARY ---\n{llm_research}"
            )

    def _sync_asknews_search(self, query: str) -> List[Any]:
        """✅ Robust search: Tries 'search_news' (v1+) then 'search_stories' (v0.7)."""
        client = self._get_asknews_client()

        if ASKNEWS_SDK_AVAILABLE:
            try:
                # 1. Try NEW SDK method (v1.0+)
                if hasattr(client.news, "search_news"):
                    response = client.news.search_news(
                        query=query,
                        n_articles=5,
                        return_type="news",
                        method="kw", # v1+ specific
                        return_story_text=True
                    )
                # 2. Try OLD SDK method (v0.7)
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
                
                # Handle response formats
                if hasattr(response, "news"):
                    return response.news
                if hasattr(response, "as_dict"):
                    return response.as_dict().get("news", [])
                
                # Fallback for dict-like response
                data = getattr(response, "data", response if isinstance(response, dict) else {})
                return data.get("news", []) if isinstance(data, dict) else []
            
            except Exception as e:
                logger.error(f"AskNews SDK search error: {e}")
                return []

        else:
            # Requests Fallback
            headers = {"Authorization": f"Bearer {client['token']}"}
            params = {
                "q": query,
                "n_articles": 5,
                "sort": "relevance",
                "return_type": "news",
                "use_neural_search": "false",
                "return_story_text": "true"
            }
            try:
                resp = requests.get(
                    "https://api.asknews.app/v1/news",
                    headers=headers,
                    params=params,
                    timeout=15
                )
                resp.raise_for_status()
                json_resp = resp.json()
                data = json_resp.get("data", json_resp)
                return data.get("news", [])
            except Exception as e:
                logger.error(f"AskNews HTTP search error: {e}")
                return []

    # --- Forecasting methods ---

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
        # Force float conversion
        try:
            val = float(pred.prediction_in_decimal)
        except (ValueError, TypeError):
            val = 0.5
            
        decimal_pred = max(0.01, min(0.99, val))
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
        unit = getattr(question, 'unit_of_measure', 'inferred')
        
        prompt = clean_indents(f"""
            You are a professional forecaster.
            Question: {question.question_text}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}
            Units: {unit}
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
        
        # ✅ FIX 2: Sanitize types before creating distribution
        # The error "< not supported between str and int" happens if p.value is a string
        clean_percentiles = []
        for p in percentile_list:
            try:
                # Strip commas or % signs just in case
                v_str = str(p.value).replace(',', '').replace('%', '').strip()
                val = float(v_str)
                clean_percentiles.append(Percentile(value=val, percentile=p.percentile))
            except (ValueError, TypeError):
                continue
                
        if not clean_percentiles:
            # Fallback to defaults if parsing totally failed
            mid = (question.lower_bound + question.upper_bound) / 2 if (question.lower_bound is not None and question.upper_bound is not None) else 0.0
            clean_percentiles = [Percentile(value=mid, percentile=0.5)]

        dist = NumericDistribution.from_question(clean_percentiles, question)
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        # Ensure bounds are treated as strings for display, but safe logic elsewhere
        low = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
        high = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
        low_msg = f"The outcome cannot be lower than {low}." if not question.open_lower_bound else f"The question creator thinks it's unlikely to be below {low}."
        high_msg = f"The outcome cannot be higher than {high}." if not question.open_upper_bound else f"The question creator thinks it's unlikely to be above {high}."
        return low_msg, high_msg

    # -----------------------------
    # Type-safe aggregation
    # -----------------------------
    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        predictions = []
        reasonings = []

        # Run 5 separate forecasts
        for _ in range(5):
            try:
                if isinstance(question, BinaryQuestion):
                    pred = await self._run_forecast_on_binary(question, research)
                elif isinstance(question, MultipleChoiceQuestion):
                    pred = await self._run_forecast_on_multiple_choice(question, research)
                elif isinstance(question, NumericQuestion):
                    pred = await self._run_forecast_on_numeric(question, research)
                else:
                    raise ValueError(f"Unsupported question type: {type(question)}")
                
                predictions.append(pred.prediction_value)
                reasonings.append(pred.reasoning)
            except Exception as e:
                # Log traceback for debugging if needed
                logger.error(f"Individual forecaster failed: {e}")
                continue

        if not predictions:
            raise RuntimeError("All 5 forecasters failed.")

        # Aggregation Logic
        final_pred = None

        if isinstance(question, BinaryQuestion):
            # Safe filtering
            numeric_preds = [p for p in predictions if isinstance(p, (int, float)) and not isinstance(p, bool)]
            if not numeric_preds:
                numeric_preds = [0.5]
            median_val = median(numeric_preds)
            final_pred = ReasonedPrediction(prediction_value=median_val, reasoning=" | ".join(reasonings))

        elif isinstance(question, MultipleChoiceQuestion):
            options = question.options
            avg_probs = {}
            for opt in options:
                option_probs = []
                for p in predictions:
                    if isinstance(p, PredictedOptionList):
                        pred_dict = {po.option_name: po.probability for po in p.predicted_options}
                        prob = pred_dict.get(opt)
                        # Ensure probability is a valid number
                        if prob is not None and isinstance(prob, (int, float)):
                            option_probs.append(float(prob))
                        else:
                            option_probs.append(0.0)
                
                if option_probs:
                    avg_probs[opt] = median(option_probs)
                else:
                    avg_probs[opt] = 0.0

            # Normalize
            total = sum(avg_probs.values())
            if total > 0:
                avg_probs = {k: v / total for k, v in avg_probs.items()}
            
            predicted_options_list = [
                PredictedOption(option_name=opt, probability=prob)
                for opt, prob in avg_probs.items()
            ]
            final_pred = ReasonedPrediction(
                prediction_value=PredictedOptionList(predicted_options=predicted_options_list),
                reasoning=" | ".join(reasonings)
            )

        elif isinstance(question, NumericQuestion):
            target_pts = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            median_percentiles = []
            
            for pt in target_pts:
                vals = []
                for p in predictions:
                    if isinstance(p, NumericDistribution):
                        for item in p.declared_percentiles:
                            # Robust check: item.value might be string or float
                            try:
                                val = float(str(item.value).replace(',',''))
                            except (ValueError, TypeError):
                                continue

                            # Normalize percentile check (handle 0.1 vs 10.0 mismatch)
                            try:
                                p_val = float(item.percentile)
                                is_match = abs(p_val - pt) < 0.01 or abs(p_val - (pt * 100)) < 1.0
                                if is_match:
                                    vals.append(val)
                            except:
                                continue
                
                # If aggregation fails, try to use midpoint of bounds as fallback
                if vals:
                    median_val = median(vals)
                else:
                    # Try to find a reasonable center
                    l = question.lower_bound or 0.0
                    u = question.upper_bound or 100.0
                    try:
                        median_val = (float(l) + float(u)) / 2.0
                    except:
                        median_val = 0.0

                median_percentiles.append(Percentile(percentile=pt, value=median_val))
            
            final_dist = NumericDistribution.from_question(median_percentiles, question)
            final_pred = ReasonedPrediction(prediction_value=final_dist, reasoning=" | ".join(reasonings))

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

    parser = argparse.ArgumentParser(description="Run FinalTournamentBot2025 (AskNews + LLMs)")
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        type=str,
        default=["ACX2026", "32916", "market-pulse-26q1", "minibench"],
    )
    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("❌ OPENROUTER_API_KEY is required")
        exit(1)

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
        exit(1)
