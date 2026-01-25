import argparse
import asyncio
import logging
import os
import textwrap
import re
import math
from datetime import datetime, date
from typing import List, Union, Dict, Any, Optional
from scipy.stats import norm

# -----------------------------
# External SDKs (with fallbacks)
# -----------------------------

try:
    from asknews_sdk import AskNewsSDK
    ASKNEWS_SDK_AVAILABLE = True
except ImportError:
    ASKNEWS_SDK_AVAILABLE = False
    import requests

try:
    from tavily import TavilyClient
    TAVILY_SDK_AVAILABLE = True
except ImportError:
    TAVILY_SDK_AVAILABLE = False

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
# Helpers
# -----------------------------
def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def median(lst: List[Union[float, int]]) -> float:
    numeric_vals = [x for x in lst if _is_num(x)]
    if not numeric_vals:
        raise ValueError("median() arg contains no numeric values")
    sorted_vals = sorted(float(x) for x in numeric_vals)
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
    m = mean(xs)
    s = stdev(xs)
    se = s / math.sqrt(len(xs)) if xs else 0.0
    z = 1.645
    lo = max(0.0, m - z * se)
    hi = min(1.0, m + z * se)
    return lo, hi

def entropy(probs: Dict[str, float]) -> float:
    e = 0.0
    for p in probs.values():
        if p > 0:
            e -= p * math.log(p)
    return e

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except Exception:
        return default

def normalize_percentile(p: Any) -> float:
    perc = safe_float(p, default=0.5)
    if perc > 1.0:
        perc = perc / 100.0
    return max(0.0, min(1.0, perc))

def build_asknews_query(question: MetaculusQuestion, max_chars: int = 397) -> str:
    q = (question.question_text or "").strip()
    bg = (question.background_info or "").strip()
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
        space_for_bg = max_chars - len(q) - 3
        if space_for_bg > 10:
            bg_part = textwrap.shorten(bg, width=space_for_bg, placeholder="…")
            return f"{q} — {bg_part}"
        return q
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
# Logging & Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("FinalTournamentBot2025")

ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_CLIENT_SECRET = os.getenv("ASKNEWS_CLIENT_SECRET")
ASKNEWS_ENABLED = bool(ASKNEWS_CLIENT_ID and ASKNEWS_CLIENT_SECRET)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_ENABLED = bool(TAVILY_API_KEY)

class FinalTournamentBot2025(ForecastBot):
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> Dict[str, str]:
        defaults = super()._llm_config_defaults()
        defaults.update({
            # Forecasters
            "forecaster_gpt": "openrouter/openai/gpt-5.2",
            "forecaster_claude": "openrouter/anthropic/claude-sonnet-4.5",
            # Parser
            "parser": "openrouter/anthropic/claude-sonnet-4.5",
            # NEW: Summarizer (factual compression only)
            "summarizer": "openrouter/anthropic/claude-sonnet-4.5",
            # NEW: Default fallback LLM
            "default_llm": "openrouter/anthropic/claude-sonnet-4.5",
        })
        return defaults

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._asknews_client = None
        logger.info("Initialized FinalTournamentBot2025 (AskNews + Tavily + Summarizer + Default LLM)")

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

    def _sync_asknews_search(self, query: str) -> List[Any]:
        client = self._get_asknews_client()
        if client is None:
            return []
        if ASKNEWS_SDK_AVAILABLE:
            try:
                method = getattr(client.news, "search_news", None) or getattr(client.news, "search_stories", None)
                if not method:
                    raise AttributeError("AskNews client lacks search method")
                response = method(
                    query=query,
                    n_articles=5,
                    return_type="news",
                    use_neural_search=False,
                    return_story_text=True,
                )
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
    # SUMMARIZER: Compress evidence (NO reasoning)
    # -----------------------------
    async def _summarize_evidence(self, raw_evidence: str) -> str:
        """
        Use LLM ONLY to compress long evidence into ≤200 words.
        Strictly factual. No interpretation.
        """
        prompt = clean_indents(f"""
            You are a factual summarizer. Your task is to condense the following evidence into a concise, neutral summary.

            RULES:
            - Include only dated facts, statistics, or direct quotes.
            - Do NOT interpret, predict, or infer.
            - Do NOT mention probabilities or outcomes.
            - Keep under 200 words.

            Evidence:
            {raw_evidence}

            Summary:
        """)
        try:
            summarizer = self.get_llm("summarizer", "llm")
            summary = await summarizer.invoke(prompt)
            return summary.strip()
        except Exception as e:
            logger.warning(f"Summarizer failed: {e}. Using raw evidence.")
            return raw_evidence  # fallback

    # -----------------------------
    # RESEARCH + SUMMARIZATION
    # -----------------------------
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            today_str = datetime.now().strftime("%Y-%m-%d")
            query = build_asknews_query(question)
            evidence_lines = [f"CURRENT DATE: {today_str}"]

            if ASKNEWS_ENABLED:
                try:
                    loop = asyncio.get_running_loop()
                    stories = await loop.run_in_executor(None, self._sync_asknews_search, query)
                    if stories:
                        evidence_lines.append("\n[RECENT NEWS]")
                        for i, story in enumerate(stories[:5]):
                            if isinstance(story, dict):
                                title = story.get("title", "Untitled")
                                text = (story.get("text") or "")[:300]
                                pub_date = story.get("publish_date", "")
                            else:
                                title = getattr(story, "title", "Untitled")
                                text = (getattr(story, "text", "") or "")[:300]
                                pub_date = getattr(story, "publish_date", "")
                            snippet = f"{i+1}. ({pub_date}) {title}: {textwrap.shorten(text, 200)}"
                            evidence_lines.append(snippet)
                    else:
                        evidence_lines.append("\n[RECENT NEWS: None found]")
                except Exception as e:
                    evidence_lines.append(f"\n[RECENT NEWS: Error - {str(e)}]")

            if TAVILY_ENABLED and TAVILY_SDK_AVAILABLE:
                try:
                    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: tavily_client.search(
                            query=query,
                            search_depth="advanced",
                            max_results=3,
                            include_answer=False,
                        )
                    )
                    results = response.get("results", [])
                    if results:
                        evidence_lines.append("\n[GENERAL FACTS]")
                        for i, res in enumerate(results[:3]):
                            content = res.get("content", "")[:300]
                            url = res.get("url", "")
                            snippet = f"{i+1}. {textwrap.shorten(content, 200)} (Source: {url})"
                            evidence_lines.append(snippet)
                    else:
                        evidence_lines.append("\n[GENERAL FACTS: None found]")
                except Exception as e:
                    evidence_lines.append(f"\n[GENERAL FACTS: Error - {str(e)}]")

            raw_evidence = "\n".join(evidence_lines)
            # NEW: Summarize to reduce noise and length
            summarized = await self._summarize_evidence(raw_evidence)
            return summarized

    # -----------------------------
    # FORECASTING (LLM used here with principles)
    # -----------------------------
    def _get_forecaster_llm(self, model_tag: str):
        try:
            return self.get_llm("forecaster_claude", "llm") if model_tag == "claude" else self.get_llm("forecaster_gpt", "llm")
        except Exception:
            logger.warning(f"Forecaster for {model_tag} unavailable. Using default LLM.")
            return self.get_llm("default_llm", "llm")

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str, run_id: int, model_tag: str) -> ReasonedPrediction[float]:
        days_to_res = (question.resolution_date - date.today()).days if question.resolution_date else "unknown"
        prompt = clean_indents(f"""
            You are a top-tier superforecaster...

            [REST OF PROMPT UNCHANGED — see previous version]
        """)
        llm = self._get_forecaster_llm(model_tag)
        reasoning = await llm.invoke(prompt)

        pred: BinaryPrediction = await structure_output(reasoning, BinaryPrediction, model=self.get_llm("parser", "llm"))
        val = safe_float(getattr(pred, "prediction_in_decimal", None), default=0.5)
        decimal_pred = max(0.01, min(0.99, float(val)))
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    # ... (keep _run_forecast_on_multiple_choice and _run_forecast_on_numeric unchanged)

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

    # ... (keep _make_prediction unchanged)

# -----------------------------
# MAIN (unchanged)
# -----------------------------
if __name__ == "__main__":
    litellm_logger = logging.getLogger("litellm")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(description="FinalTournamentBot2025: Evidence-based forecasting with summarizer")
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
        raise SystemExit(1)
