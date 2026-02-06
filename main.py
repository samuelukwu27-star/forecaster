import argparse
import asyncio
import logging
import os
import textwrap
import re
import math
from datetime import datetime
from typing import List, Union, Dict, Any, Optional, Tuple

# Optional: Tavily
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

# AskNews integration
try:
    from asknews_sdk import AskNewsSDK
    ASKNEWS_SDK_AVAILABLE = True
except ImportError:
    ASKNEWS_SDK_AVAILABLE = False
    import requests  # type: ignore

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
# Helpers: robust stats + parsing
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

def ci90(xs: List[float]) -> Tuple[float, float]:
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

def safe_float(x: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s:
            return default
        s = s.replace(",", "").replace("%", "").strip()
        return float(s)
    except Exception:
        return default

def normalize_percentile(p: Any) -> float:
    perc = safe_float(p, default=0.5)
    if perc is None:
        return 0.5
    if perc > 1.0:
        perc = perc / 100.0
    if perc < 0.0:
        perc = 0.0
    if perc > 1.0:
        perc = 1.0
    return float(perc)

def clamp01(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, float(p)))

_PERCENT_RE = re.compile(r"(?i)\bprob(?:ability)?\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%")
_DEC_RE = re.compile(r"(?i)\bdecimal\s*:\s*([0-9]*\.?[0-9]+)\b")

def extract_binary_prob_from_text(text: str) -> Optional[float]:
    """
    Fallback extractor for binary probs.
    Accepts:
      - "Probability: 63%"
      - "Decimal: 0.63"
    Returns decimal in [0,1] or None.
    """
    if not text:
        return None

    m = _PERCENT_RE.search(text)
    if m:
        pct = safe_float(m.group(1), default=None)
        if pct is None:
            return None
        return clamp01(pct / 100.0, 0.0, 1.0)

    m = _DEC_RE.search(text)
    if m:
        dec = safe_float(m.group(1), default=None)
        if dec is None:
            return None
        return clamp01(dec, 0.0, 1.0)

    # Last resort: find a standalone percent like "63%"
    m2 = re.search(r"(?<!\d)([0-9]{1,3}(?:\.[0-9]+)?)\s*%", text)
    if m2:
        pct = safe_float(m2.group(1), default=None)
        if pct is None:
            return None
        return clamp01(pct / 100.0, 0.0, 1.0)

    return None

def build_indexed_options(options: List[str]) -> List[str]:
    return [f"{i+1}) {opt}" for i, opt in enumerate(options)]

def extract_indexed_mc_probs(text: str, n_options: int) -> Dict[int, float]:
    """
    Fallback extractor for multiple choice:
      "1: 12%" or "1) 12%" or "Option 1: 12%"
    Returns dict {index(1..n): prob_decimal}
    """
    out: Dict[int, float] = {}
    if not text:
        return out
    patterns = [
        re.compile(r"(?i)\b(?:option\s*)?(\d{1,2})\s*[:\)\-]\s*([0-9]+(?:\.[0-9]+)?)\s*%"),
        re.compile(r"(?i)\b(\d{1,2})\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*%"),
    ]
    for pat in patterns:
        for m in pat.finditer(text):
            idx = int(m.group(1))
            if 1 <= idx <= n_options:
                pct = safe_float(m.group(2), default=None)
                if pct is None:
                    continue
                out[idx] = float(pct) / 100.0
    return out

def extract_numeric_percentiles(text: str, targets: List[float]) -> Dict[float, float]:
    """
    Fallback extractor for numeric percentiles:
      "Percentile 10: X" or "P10: X"
    Returns dict {target_percentile_decimal: value}
    """
    out: Dict[float, float] = {}
    if not text:
        return out

    # capture 10/20/40/60/80/90 etc
    for pt in targets:
        pct_int = int(round(pt * 100))
        pats = [
            re.compile(rf"(?i)\bpercentile\s*{pct_int}\s*:\s*([-+]?[0-9,]*\.?[0-9]+)"),
            re.compile(rf"(?i)\bp\s*{pct_int}\s*:\s*([-+]?[0-9,]*\.?[0-9]+)"),
            re.compile(rf"(?i)\bp{pct_int}\s*=\s*([-+]?[0-9,]*\.?[0-9]+)"),
        ]
        for pat in pats:
            m = pat.search(text)
            if m:
                v = safe_float(m.group(1), default=None)
                if v is not None:
                    out[pt] = float(v)
                    break
    return out

# -----------------------------
# ASKNEWS QUERY BUILDER
# -----------------------------
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
# Tavily Integration
# -----------------------------
def _get_tavily_client() -> Optional["TavilyClient"]:
    if not TAVILY_AVAILABLE:
        return None
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None
    return TavilyClient(api_key=api_key)

def _sync_tavily_search(query: str, max_results: int = 5) -> List[str]:
    client = _get_tavily_client()
    if client is None:
        return []
    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            include_answer=False,
            max_results=max_results,
            include_raw_content=False,
        )
        results = response.get("results", [])
        snippets = []
        for i, res in enumerate(results[:max_results]):
            title = res.get("title", "Untitled")
            content = (res.get("content", "") or "")[:500]
            snippet = f"[T{i+1}] {title}: {textwrap.shorten(content, width=240, placeholder='…')}"
            snippets.append(snippet)
        return snippets
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return []

# -----------------------------
# Logging + env
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
    logger.warning("ASKNEWS_CLIENT_ID/ASKNEWS_CLIENT_SECRET not set — AskNews disabled.")

TAVILY_ENABLED = TAVILY_AVAILABLE and bool(os.getenv("TAVILY_API_KEY"))
if not TAVILY_ENABLED:
    logger.warning("TAVILY_API_KEY not set — Tavily disabled.")

# -----------------------------
# Bot Class (2-model, 2 principles)
# -----------------------------
class FinalTournamentBot2025(ForecastBot):
    """
    Two-model setup:
      - GPT-5.2: primary forecaster
      - Claude Sonnet 4.5: adversarial checker (still outputs a forecast)

    Two core Good Judgment principles enforced in prompts:
      1) Outside view / base rates first.
      2) Consider-the-opposite (steelman the opposite side) before finalizing.

    Aggregation:
      - Binary: weighted blend 0.7 GPT / 0.3 Claude
      - Multiple choice: weighted blend per option, renormalize
      - Numeric: weighted blend of percentile values
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    def _llm_config_defaults(self) -> Dict[str, str]:
        defaults = super()._llm_config_defaults()
        defaults.update({
            "researcher_gpt": "openrouter/openai/gpt-5.2",
            "researcher_claude": "openrouter/anthropic/claude-sonnet-4.5",
            "forecaster_gpt": "openrouter/openai/gpt-5.2",
            "forecaster_claude": "openrouter/anthropic/claude-sonnet-4.5",
            "parser": "openrouter/anthropic/claude-sonnet-4.5",
        })
        return defaults

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._asknews_client = None
        logger.info("Initialized FinalTournamentBot2025 (AskNews + Tavily + 2-model high-score intent)")
        # Only 2 runs total
        self._run_schedule = ["gpt", "claude"]

        # Drop accounting
        self._drop_counts: Dict[str, int] = {}
        self._drop_counts_by_model: Dict[str, Dict[str, int]] = {"gpt": {}, "claude": {}}

    def _inc_drop(self, model_tag: str, reason: str) -> None:
        self._drop_counts[reason] = self._drop_counts.get(reason, 0) + 1
        d = self._drop_counts_by_model.get(model_tag, {})
        d[reason] = d.get(reason, 0) + 1
        self._drop_counts_by_model[model_tag] = d

    # -----------------------------
    # AskNews Client
    # -----------------------------
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
    # Research (concise, scorer-friendly)
    # -----------------------------
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            today_str = datetime.now().strftime("%Y-%m-%d")
            query = build_asknews_query(question)

            # AskNews + Tavily snippets (optional)
            asknews_summary = "[AskNews disabled]"
            if ASKNEWS_ENABLED:
                try:
                    loop = asyncio.get_running_loop()
                    stories = await loop.run_in_executor(None, self._sync_asknews_search, query)
                    if stories:
                        snippets = []
                        for i, story in enumerate(stories[:5]):
                            if isinstance(story, dict):
                                title = story.get("title", "Untitled")
                                text = (story.get("text") or "")[:500]
                            else:
                                title = getattr(story, "title", "Untitled")
                                text = (getattr(story, "text", "") or "")[:500]
                            snippets.append(f"[A{i+1}] {title}: {textwrap.shorten(text, width=220, placeholder='…')}")
                        asknews_summary = "\n".join(snippets)
                    else:
                        asknews_summary = "[AskNews: No recent stories found]"
                except Exception as e:
                    asknews_summary = f"[AskNews error: {str(e)}]"
                    logger.error(f"AskNews research failed: {e}")

            tavily_summary = "[Tavily disabled]"
            if TAVILY_ENABLED:
                try:
                    loop = asyncio.get_running_loop()
                    tavily_snippets = await loop.run_in_executor(None, _sync_tavily_search, query)
                    tavily_summary = "\n".join(tavily_snippets) if tavily_snippets else "[Tavily: No results found]"
                except Exception as e:
                    tavily_summary = f"[Tavily error: {str(e)}]"
                    logger.error(f"Tavily research failed: {e}")

            # Single structured research pass (GPT) + brief Claude check for missing angle
            research_prompt = clean_indents(f"""
                You are helping a forecaster maximize scoring rules (log/Brier). Be concise and decision-relevant.
                Two principles:
                1) Outside view/base rates first.
                2) Consider-the-opposite: strongest case for the other side.

                Question: {question.question_text}
                Resolution Criteria: {question.resolution_criteria}
                Fine Print: {question.fine_print}

                Output <= 200 words in this exact structure:
                - Key facts (dated):
                - Outside view / base rate:
                - Main drivers to watch:
                - Strongest case for opposite outcome:
                - What would change your mind:
            """)

            try:
                llm_gpt = self.get_llm("researcher_gpt", "llm")
                gpt_research = await llm_gpt.invoke(research_prompt)
            except Exception as e:
                gpt_research = f"[LLM research (GPT) failed: {str(e)}]"

            try:
                llm_claude = self.get_llm("researcher_claude", "llm")
                claude_gap_check = await llm_claude.invoke(clean_indents(f"""
                    You are the adversarial checker. Find missing considerations that would materially change the forecast.
                    Keep <= 120 words. Bullet points only.

                    Question: {question.question_text}
                    Resolution Criteria: {question.resolution_criteria}

                    Research draft:
                    {gpt_research}

                    Output:
                    - Missing consideration(s):
                    - Potential direction of update (up/down) and why:
                """))
            except Exception as e:
                claude_gap_check = f"[LLM gap-check (Claude) failed: {str(e)}]"

            return (
                f"--- LIVE NEWS (as of {today_str}) ---\n{asknews_summary}\n\n"
                f"--- WEB SEARCH SNIPPETS ---\n{tavily_summary}\n\n"
                f"--- STRUCTURED RESEARCH (GPT-5.2) ---\n{gpt_research}\n\n"
                f"--- ADVERSARIAL GAP CHECK (SONNET-4.5) ---\n{claude_gap_check}"
            )

    # -----------------------------
    # Forecast invocation helpers (retry + fallbacks)
    # -----------------------------
    async def _invoke_llm(self, model_name: str, prompt: str) -> str:
        llm = self.get_llm(model_name, "llm")
        return await llm.invoke(prompt)

    async def _invoke_with_format_retry(self, model_name: str, prompt: str, format_spec: str) -> str:
        """
        One retry if the first response likely violates format.
        """
        try:
            return await self._invoke_llm(model_name, prompt)
        except Exception:
            raise

    async def _parse_binary(self, raw: str, model_tag: str) -> Optional[float]:
        # Structured parse attempt
        try:
            pred: BinaryPrediction = await structure_output(
                text_to_structure=raw,
                output_type=BinaryPrediction,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )
            val = safe_float(getattr(pred, "prediction_in_decimal", None), default=None)
            if val is not None:
                return clamp01(float(val))
        except Exception:
            self._inc_drop(model_tag, "parse_error_binary_structured")

        # Fallback parse
        val2 = extract_binary_prob_from_text(raw)
        if val2 is None:
            self._inc_drop(model_tag, "parse_error_binary_fallback")
            return None
        return clamp01(float(val2))

    async def _parse_mc(self, raw: str, question: MultipleChoiceQuestion, model_tag: str) -> Optional[Dict[str, float]]:
        options = list(question.options)
        n = len(options)

        # Structured parse attempt
        try:
            pred: PredictedOptionList = await structure_output(
                text_to_structure=raw,
                output_type=PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                additional_instructions=f"Valid options: {options}. Prefer interpreting numbered options if present.",
                num_validation_samples=self._structure_output_validation_samples,
            )
            pred_dict = {}
            for po in pred.predicted_options:
                name = str(po.option_name).strip()
                prob = po.probability
                if _is_num(prob):
                    pred_dict[name] = float(prob)
            # Map to canonical options (best-effort)
            out: Dict[str, float] = {}
            for opt in options:
                if opt in pred_dict:
                    out[opt] = pred_dict[opt]
                else:
                    # casefold match
                    opt_cf = opt.casefold()
                    for k, v in pred_dict.items():
                        if k.casefold() == opt_cf:
                            out[opt] = v
                            break
            if out:
                return out
        except Exception:
            self._inc_drop(model_tag, "parse_error_mc_structured")

        # Fallback: indexed extraction
        idx_probs = extract_indexed_mc_probs(raw, n)
        if not idx_probs:
            self._inc_drop(model_tag, "parse_error_mc_fallback")
            return None
        out2: Dict[str, float] = {}
        for i in range(1, n + 1):
            if i in idx_probs:
                out2[options[i - 1]] = float(idx_probs[i])
        if not out2:
            self._inc_drop(model_tag, "parse_error_mc_fallback_empty")
            return None
        return out2

    async def _parse_numeric(self, raw: str, question: NumericQuestion, model_tag: str) -> Optional[NumericDistribution]:
        targets = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

        # Structured parse attempt
        try:
            percentile_list: List[Percentile] = await structure_output(
                text_to_structure=raw,
                output_type=list[Percentile],
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )
            clean_percentiles: List[Percentile] = []
            for p in percentile_list:
                val = safe_float(getattr(p, "value", None), default=None)
                if val is None:
                    continue
                perc = normalize_percentile(getattr(p, "percentile", 0.5))
                clean_percentiles.append(Percentile(value=float(val), percentile=float(perc)))
            if clean_percentiles:
                clean_percentiles.sort(key=lambda x: float(x.percentile))
                for i in range(1, len(clean_percentiles)):
                    if clean_percentiles[i].value < clean_percentiles[i - 1].value:
                        clean_percentiles[i].value = clean_percentiles[i - 1].value
                return NumericDistribution.from_question(clean_percentiles, question)
        except Exception:
            self._inc_drop(model_tag, "parse_error_numeric_structured")

        # Fallback extraction
        extracted = extract_numeric_percentiles(raw, targets)
        if not extracted:
            self._inc_drop(model_tag, "parse_error_numeric_fallback")
            return None

        pts: List[Percentile] = []
        for pt in targets:
            if pt in extracted:
                pts.append(Percentile(percentile=pt, value=float(extracted[pt])))

        pts.sort(key=lambda x: float(x.percentile))
        for i in range(1, len(pts)):
            if pts[i].value < pts[i - 1].value:
                pts[i].value = pts[i - 1].value

        return NumericDistribution.from_question(pts, question)

    def _create_upper_and_lower_bound_messages(self, question: NumericQuestion) -> Tuple[str, str]:
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
    # Core forecasting (GPT primary + Claude adversarial checker)
    # -----------------------------
    def _binary_prompt(self, question: BinaryQuestion, research: str, role: str) -> str:
        # Two Good Judgment principles baked in, plus scoring intent.
        return clean_indents(f"""
            You are forecasting to maximize proper scoring rules (log score / Brier). Be calibrated and decisive.
            Apply exactly two principles:
            1) Outside view / base rates first.
            2) Consider-the-opposite: strongest case for the other side before finalizing.

            Role: {role}
            Today: {datetime.now().strftime('%Y-%m-%d')}

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}

            Research:
            {research}

            Write 6-10 bullets total, then output EXACTLY these two lines at the end:
            Probability: ZZ%
            Decimal: 0.ZZ
        """)

    def _mc_prompt(self, question: MultipleChoiceQuestion, research: str, role: str) -> str:
        options = list(question.options)
        indexed = build_indexed_options(options)
        return clean_indents(f"""
            You are forecasting to maximize proper scoring rules. Be calibrated and decisive.
            Apply exactly two principles:
            1) Outside view / base rates first.
            2) Consider-the-opposite: strongest case for the leading alternative.

            Role: {role}
            Today: {datetime.now().strftime('%Y-%m-%d')}

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}

            Options (numbered, MUST use these numbers in your final lines):
            {chr(10).join(indexed)}

            Research:
            {research}

            Write 6-10 bullets total, then output EXACTLY n lines at the end:
            1: XX%
            2: XX%
            ...
            {len(options)}: XX%
            (These must sum to 100%.)
        """)

    def _numeric_prompt(self, question: NumericQuestion, research: str, role: str) -> str:
        low_msg, high_msg = self._create_upper_and_lower_bound_messages(question)
        unit = getattr(question, "unit_of_measure", "inferred")
        return clean_indents(f"""
            You are forecasting to maximize proper scoring rules. Be calibrated and decisive.
            Apply exactly two principles:
            1) Outside view / base rates first.
            2) Consider-the-opposite: strongest case for a much-lower or much-higher outcome.

            Role: {role}
            Today: {datetime.now().strftime('%Y-%m-%d')}

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}
            Units: {unit}

            Bounds guidance:
            {low_msg}
            {high_msg}

            Research:
            {research}

            Write 6-10 bullets total, then output EXACTLY these lines at the end:
            Percentile 10: X
            Percentile 20: X
            Percentile 40: X
            Percentile 60: X
            Percentile 80: X
            Percentile 90: X
        """)

    async def _run_binary_role(self, question: BinaryQuestion, research: str, model_tag: str, role: str) -> ReasonedPrediction[float]:
        model_name = f"forecaster_{model_tag}"
        prompt = self._binary_prompt(question, research, role)

        try:
            raw = await self._invoke_with_format_retry(model_name, prompt, "Probability/Decimal lines")
        except Exception as e:
            self._inc_drop(model_tag, "llm_error_binary")
            raise e

        val = await self._parse_binary(raw, model_tag=model_tag)
        if val is None:
            # One short reprompt focused on format only
            reprompt = clean_indents(f"""
                Output ONLY the final two lines in this exact format (no other text):
                Probability: ZZ%
                Decimal: 0.ZZ
            """)
            try:
                raw2 = await self._invoke_llm(model_name, reprompt)
                val = await self._parse_binary(raw2, model_tag=model_tag)
                raw = raw + "\n\n[FORMAT_RETRY]\n" + raw2
            except Exception:
                self._inc_drop(model_tag, "retry_failed_binary")

        if val is None:
            self._inc_drop(model_tag, "invalid_values_binary")
            val = 0.5

        return ReasonedPrediction(prediction_value=clamp01(val), reasoning=raw)

    async def _run_mc_role(self, question: MultipleChoiceQuestion, research: str, model_tag: str, role: str) -> ReasonedPrediction[PredictedOptionList]:
        model_name = f"forecaster_{model_tag}"
        prompt = self._mc_prompt(question, research, role)

        try:
            raw = await self._invoke_with_format_retry(model_name, prompt, "Indexed option lines")
        except Exception as e:
            self._inc_drop(model_tag, "llm_error_mc")
            raise e

        probs = await self._parse_mc(raw, question, model_tag=model_tag)
        if probs is None:
            reprompt = clean_indents(f"""
                Output ONLY the final probability lines using option numbers and percents.
                Example:
                1: 12%
                2: 34%
                ...
                Must sum to 100%.
            """)
            try:
                raw2 = await self._invoke_llm(model_name, reprompt)
                probs = await self._parse_mc(raw2, question, model_tag=model_tag)
                raw = raw + "\n\n[FORMAT_RETRY]\n" + raw2
            except Exception:
                self._inc_drop(model_tag, "retry_failed_mc")

        if probs is None:
            self._inc_drop(model_tag, "invalid_values_mc")
            # uniform fallback
            options = list(question.options)
            u = 1.0 / max(1, len(options))
            probs = {opt: u for opt in options}

        # Build PredictedOptionList
        predicted_options_list = [
            PredictedOption(option_name=opt, probability=float(p))
            for opt, p in probs.items()
        ]
        return ReasonedPrediction(
            prediction_value=PredictedOptionList(predicted_options=predicted_options_list),
            reasoning=raw,
        )

    async def _run_numeric_role(self, question: NumericQuestion, research: str, model_tag: str, role: str) -> ReasonedPrediction[NumericDistribution]:
        model_name = f"forecaster_{model_tag}"
        prompt = self._numeric_prompt(question, research, role)

        try:
            raw = await self._invoke_with_format_retry(model_name, prompt, "Percentile lines")
        except Exception as e:
            self._inc_drop(model_tag, "llm_error_numeric")
            raise e

        dist = await self._parse_numeric(raw, question, model_tag=model_tag)
        if dist is None:
            reprompt = clean_indents(f"""
                Output ONLY these lines (no other text):
                Percentile 10: X
                Percentile 20: X
                Percentile 40: X
                Percentile 60: X
                Percentile 80: X
                Percentile 90: X
            """)
            try:
                raw2 = await self._invoke_llm(model_name, reprompt)
                dist = await self._parse_numeric(raw2, question, model_tag=model_tag)
                raw = raw + "\n\n[FORMAT_RETRY]\n" + raw2
            except Exception:
                self._inc_drop(model_tag, "retry_failed_numeric")

        if dist is None:
            self._inc_drop(model_tag, "invalid_values_numeric")
            # midpoint fallback
            try:
                l = float(question.lower_bound) if question.lower_bound is not None else 0.0
            except Exception:
                l = 0.0
            try:
                u = float(question.upper_bound) if question.upper_bound is not None else 100.0
            except Exception:
                u = 100.0
            mid = (l + u) / 2.0
            dist = NumericDistribution.from_question([Percentile(value=mid, percentile=0.5)], question)

        return ReasonedPrediction(prediction_value=dist, reasoning=raw)

    # -----------------------------
    # Abstract method implementations (kept, but GPT-only)
    # -----------------------------
    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        # For single-run fallback behavior, keep GPT primary
        return await self._run_binary_role(question, research, model_tag="gpt", role="PRIMARY")

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        return await self._run_mc_role(question, research, model_tag="gpt", role="PRIMARY")

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        return await self._run_numeric_role(question, research, model_tag="gpt", role="PRIMARY")

    # -----------------------------
    # Custom aggregation (2 forecasters only)
    # -----------------------------
    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        # Guard prediction too (reduces provider rate-limit drops)
        async with self._concurrency_limiter:
            preds: List[Any] = []
            reasonings: List[str] = []

            # Run GPT primary
            try:
                if isinstance(question, BinaryQuestion):
                    gpt_pred = await self._run_binary_role(question, research, "gpt", "PRIMARY")
                elif isinstance(question, MultipleChoiceQuestion):
                    gpt_pred = await self._run_mc_role(question, research, "gpt", "PRIMARY")
                elif isinstance(question, NumericQuestion):
                    gpt_pred = await self._run_numeric_role(question, research, "gpt", "PRIMARY")
                else:
                    raise ValueError(f"Unsupported question type: {type(question)}")
                preds.append(gpt_pred.prediction_value)
                reasonings.append("[GPT_PRIMARY]\n" + gpt_pred.reasoning)
            except Exception as e:
                logger.error(f"GPT primary failed: {e}")

            # Run Claude adversarial checker (independent forecast, adversarial stance)
            try:
                if isinstance(question, BinaryQuestion):
                    cl_pred = await self._run_binary_role(question, research, "claude", "ADVERSARIAL_CHECKER")
                elif isinstance(question, MultipleChoiceQuestion):
                    cl_pred = await self._run_mc_role(question, research, "claude", "ADVERSARIAL_CHECKER")
                elif isinstance(question, NumericQuestion):
                    cl_pred = await self._run_numeric_role(question, research, "claude", "ADVERSARIAL_CHECKER")
                else:
                    raise ValueError(f"Unsupported question type: {type(question)}")
                preds.append(cl_pred.prediction_value)
                reasonings.append("[CLAUDE_CHECKER]\n" + cl_pred.reasoning)
            except Exception as e:
                logger.error(f"Claude checker failed: {e}")

            if not preds:
                raise RuntimeError("All forecasters failed.")

            w_gpt, w_claude = 0.70, 0.30

            # BINARY: weighted blend
            if isinstance(question, BinaryQuestion):
                # If one missing, fall back to the one present
                g = float(preds[0]) if len(preds) >= 1 and _is_num(preds[0]) else None
                c = float(preds[1]) if len(preds) >= 2 and _is_num(preds[1]) else None

                if g is not None and c is not None:
                    final = clamp01(w_gpt * g + w_claude * c)
                    numeric_preds = [g, c]
                elif g is not None:
                    final = clamp01(g)
                    numeric_preds = [g]
                elif c is not None:
                    final = clamp01(c)
                    numeric_preds = [c]
                else:
                    final = 0.5
                    numeric_preds = [0.5]

                med = median(numeric_preds)
                m = mean(numeric_preds)
                s = stdev(numeric_preds)
                lo, hi = ci90(numeric_preds)
                stats_line = (
                    f"[stats] n={len(numeric_preds)} mean={m:.3f} median={med:.3f} "
                    f"sd={s:.3f} ci90=({lo:.3f},{hi:.3f}) agg=weighted(0.7/0.3)"
                )
                return ReasonedPrediction(
                    prediction_value=final,
                    reasoning=stats_line + "\n\n" + "\n\n---\n\n".join(reasonings),
                )

            # MULTIPLE CHOICE: weighted blend per option, renormalize
            if isinstance(question, MultipleChoiceQuestion):
                options = list(question.options)

                # Convert PredictedOptionList to dict
                def pol_to_dict(pol: Any) -> Dict[str, float]:
                    out: Dict[str, float] = {}
                    if isinstance(pol, PredictedOptionList):
                        for po in pol.predicted_options:
                            if _is_num(po.probability):
                                out[str(po.option_name).strip()] = float(po.probability)
                    return out

                g_dict = pol_to_dict(preds[0]) if len(preds) >= 1 else {}
                c_dict = pol_to_dict(preds[1]) if len(preds) >= 2 else {}

                blended: Dict[str, float] = {}
                for opt in options:
                    gv = g_dict.get(opt)
                    cv = c_dict.get(opt)
                    if gv is None and cv is None:
                        blended[opt] = 1e-6
                    elif gv is None:
                        blended[opt] = float(cv)
                    elif cv is None:
                        blended[opt] = float(gv)
                    else:
                        blended[opt] = w_gpt * float(gv) + w_claude * float(cv)

                total = sum(blended.values())
                if total <= 0:
                    u = 1.0 / max(1, len(options))
                    blended = {opt: u for opt in options}
                else:
                    blended = {k: v / total for k, v in blended.items()}

                ent = entropy(blended)
                stats_line = f"[stats] n_models={len(preds)} entropy={ent:.3f} agg=weighted(0.7/0.3)"
                predicted_options_list = [
                    PredictedOption(option_name=opt, probability=float(prob))
                    for opt, prob in blended.items()
                ]
                return ReasonedPrediction(
                    prediction_value=PredictedOptionList(predicted_options=predicted_options_list),
                    reasoning=stats_line + "\n\n" + "\n\n---\n\n".join(reasonings),
                )

            # NUMERIC: weighted blend of percentile values
            if isinstance(question, NumericQuestion):
                targets = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

                def dist_to_map(d: Any) -> Dict[float, float]:
                    out: Dict[float, float] = {}
                    if isinstance(d, NumericDistribution):
                        for item in d.declared_percentiles:
                            perc = normalize_percentile(getattr(item, "percentile", None))
                            val = safe_float(getattr(item, "value", None), default=None)
                            if val is None:
                                continue
                            out[perc] = float(val)
                    return out

                g_map = dist_to_map(preds[0]) if len(preds) >= 1 else {}
                c_map = dist_to_map(preds[1]) if len(preds) >= 2 else {}

                blended_pts: List[Percentile] = []
                for pt in targets:
                    gv = None
                    cv = None
                    # match closest declared percentile to pt
                    if g_map:
                        gv = min(g_map.items(), key=lambda kv: abs(kv[0] - pt))[1] if g_map else None
                    if c_map:
                        cv = min(c_map.items(), key=lambda kv: abs(kv[0] - pt))[1] if c_map else None

                    if gv is None and cv is None:
                        # fallback midpoint
                        try:
                            l = float(question.lower_bound) if question.lower_bound is not None else 0.0
                        except Exception:
                            l = 0.0
                        try:
                            u = float(question.upper_bound) if question.upper_bound is not None else 100.0
                        except Exception:
                            u = 100.0
                        v = (l + u) / 2.0
                    elif gv is None:
                        v = float(cv)
                    elif cv is None:
                        v = float(gv)
                    else:
                        v = w_gpt * float(gv) + w_claude * float(cv)

                    blended_pts.append(Percentile(percentile=pt, value=float(v)))

                blended_pts.sort(key=lambda x: float(x.percentile))
                for i in range(1, len(blended_pts)):
                    if blended_pts[i].value < blended_pts[i - 1].value:
                        blended_pts[i].value = blended_pts[i - 1].value

                p10 = next((p.value for p in blended_pts if abs(float(p.percentile) - 0.1) < 1e-9), None)
                p90 = next((p.value for p in blended_pts if abs(float(p.percentile) - 0.9) < 1e-9), None)
                spread = (p90 - p10) if (p10 is not None and p90 is not None) else float("nan")
                stats_line = f"[stats] n_models={len(preds)} p10={float(p10):.3f} p90={float(p90):.3f} spread={float(spread):.3f} agg=weighted(0.7/0.3)"

                final_dist = NumericDistribution.from_question(blended_pts, question)
                return ReasonedPrediction(
                    prediction_value=final_dist,
                    reasoning=stats_line + "\n\n" + "\n\n---\n\n".join(reasonings),
                )

            # Fallback for unknown types
            return ReasonedPrediction(prediction_value=preds[0], reasoning="\n\n---\n\n".join(reasonings))

    
    def log_internal_drop_stats(self) -> None:
        if not self._drop_counts:
            return
        logger.info(f"[drops] totals={self._drop_counts}")
        logger.info(f"[drops] by_model={self._drop_counts_by_model}")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    litellm_logger = logging.getLogger("litellm")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(description="Run FinalTournamentBot2025 (AskNews + Tavily + 2-model high-score intent)")
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
        predictions_per_research_report=2,  
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
        # optional diagnostics:
        bot.log_internal_drop_stats()
        logger.info("✅ FinalTournamentBot2025 run completed.")
    except Exception as e:
        logger.error(f"❌ Fatal error during execution: {e}")
        raise SystemExit(1)
