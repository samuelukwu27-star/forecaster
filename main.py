import argparse
import asyncio
import logging
import os
import textwrap
import re
import math
from datetime import datetime
from typing import List, Union, Dict, Any, Optional, Tuple

# -----------------------------
# Optional: Tavily (REQUIRED at runtime)
# -----------------------------
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

# -----------------------------
# Optional: yfinance (for Market Pulse)
# -----------------------------
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# -----------------------------
# AskNews integration (optional but supported)
# -----------------------------
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

# ============================================================
# Helpers: robust stats + parsing
# ============================================================
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
    if perc is None: return 0.5
    if perc > 1.0: perc = perc / 100.0
    if perc < 0.0: perc = 0.0
    if perc > 1.0: perc = 1.0
    return float(perc)

def clamp01(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, float(p)))

_PERCENT_RE = re.compile(r"(?i)\bprob(?:ability)?\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%")
_DEC_RE = re.compile(r"(?i)\bdecimal\s*:\s*([0-9]*\.?[0-9]+)\b")

def extract_binary_prob_from_text(text: str) -> Optional[float]:
    if not text: return None
    m = _PERCENT_RE.search(text)
    if m:
        pct = safe_float(m.group(1), default=None)
        if pct is not None: return clamp01(pct / 100.0, 0.0, 1.0)
    m = _DEC_RE.search(text)
    if m:
        dec = safe_float(m.group(1), default=None)
        if dec is not None: return clamp01(dec, 0.0, 1.0)
    m2 = re.search(r"(?<!\d)([0-9]{1,3}(?:\.[0-9]+)?)\s*%", text)
    if m2:
        pct = safe_float(m2.group(1), default=None)
        if pct is not None: return clamp01(pct / 100.0, 0.0, 1.0)
    return None

def build_indexed_options(options: List[str]) -> List[str]:
    return [f"{i+1}) {opt}" for i, opt in enumerate(options)]

def extract_indexed_mc_probs(text: str, n_options: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if not text: return out
    patterns = [
        re.compile(r"(?i)\b(?:option\s*)?(\d{1,2})\s*[:\)\-]\s*([0-9]+(?:\.[0-9]+)?)\s*%"),
        re.compile(r"(?i)\b(\d{1,2})\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*%"),
    ]
    for pat in patterns:
        for m in pat.finditer(text):
            idx = int(m.group(1))
            if 1 <= idx <= n_options:
                pct = safe_float(m.group(2), default=None)
                if pct is not None: out[idx] = float(pct) / 100.0
    return out

def extract_numeric_percentiles(text: str, targets: List[float]) -> Dict[float, float]:
    out: Dict[float, float] = {}
    if not text: return out
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

# ============================================================
# EXTREMIZATION ENGINE (Minibench + Tail Fattening)
# ============================================================
def _logit(p: float) -> float:
    p = clamp01(p, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def extremize_binary(p: float, k: float) -> float:
    if not _is_num(p) or not _is_num(k) or k <= 0 or abs(k - 1.0) < 1e-12:
        return float(p)
    return clamp01(_sigmoid(_logit(float(p)) * float(k)))

def extremize_mc(probs: Dict[str, float], k: float) -> Dict[str, float]:
    if not probs or not _is_num(k) or k <= 0 or abs(k - 1.0) < 1e-12:
        s = sum(max(0.0, float(v)) for v in probs.values())
        if s > 0: return {a: max(0.0, float(v)) / s for a, v in probs.items()}
        return {a: 1.0 / len(probs) for a in probs}

    powered: Dict[str, float] = {a: max(0.0, float(v)) ** float(k) for a, v in probs.items()}
    s2 = sum(powered.values())
    return {a: v / s2 for a, v in powered.items()} if s2 > 0 else {a: 1.0 / len(probs) for a in probs}

def apply_tail_fattening(pts: List[Percentile], factor: float = 1.15) -> List[Percentile]:
    """Widens the P10 and P90 distribution tails relative to the median."""
    p50_val = None
    for p in pts:
        if abs(float(p.percentile) - 0.5) < 1e-9:
            p50_val = p.value
            break
            
    if p50_val is None:
        l_val, r_val = None, None
        for p in pts:
            if float(p.percentile) < 0.5: l_val = p.value
            if float(p.percentile) > 0.5 and r_val is None: r_val = p.value
        if l_val is not None and r_val is not None:
            p50_val = (l_val + r_val) / 2.0
        else:
            return pts 

    for p in pts:
        pct = float(p.percentile)
        if pct < 0.5:
            p.value = p50_val - (p50_val - p.value) * factor
        elif pct > 0.5:
            p.value = p50_val + (p.value - p50_val) * factor

    pts.sort(key=lambda x: float(x.percentile))
    for i in range(1, len(pts)):
        if pts[i].value < pts[i - 1].value:
            pts[i].value = pts[i - 1].value
    return pts

# --- MINIBENCH LOGIC ---
MINIBENCH_K_BASE      = 5.0
MINIBENCH_K_AGREE     = 1.0
MINIBENCH_K_RESEARCH  = 1.0
MINIBENCH_K_MAX       = 7.0
MINIBENCH_K_MC        = 3.5
MINIBENCH_GATE_LO     = 0.40
MINIBENCH_GATE_HI     = 0.60
MINIBENCH_GATE_AMP    = 1.5
MINIBENCH_CONV_LO     = 0.44
MINIBENCH_CONV_HI     = 0.52
MINIBENCH_CONV_POS    = 0.82
MINIBENCH_CONV_NEG    = 0.18

_CONVICTION_RE = re.compile(
    r"(?i)\b(confirmed|confirmed\s+by|officially|announced|signed|passed|enacted|"
    r"launched|deployed|released|completed|achieved|won|elected|appointed|"
    r"definitively|conclusively|clearly|undeniably|certainly|already\s+has|"
    r"has\s+already|is\s+now|are\s+now|have\s+now|did\s+not|never\s+happened|"
    r"no\s+evidence|ruled\s+out|impossible\s+by)\b"
)

def _research_is_strong(research: str) -> bool:
    return len(_CONVICTION_RE.findall(research or "")) >= 2

def _agents_agree(g_val: Optional[float], c_val: Optional[float]) -> bool:
    if g_val is None or c_val is None: return False
    return (g_val > 0.5 and c_val > 0.5) or (g_val < 0.5 and c_val < 0.5)

def minibench_extremize_binary(
    blend: float, g_val: Optional[float], c_val: Optional[float], research: str
) -> Tuple[float, float, str]:
    agree   = _agents_agree(g_val, c_val)
    strong  = _research_is_strong(research)
    in_zone = MINIBENCH_CONV_LO <= blend <= MINIBENCH_CONV_HI

    if in_zone and agree and strong:
        pos    = blend > 0.50
        result = MINIBENCH_CONV_POS if pos else MINIBENCH_CONV_NEG
        return result, MINIBENCH_K_MAX, f"T5({'pos' if pos else 'neg'} {blend:.3f}->{result:.3f})"

    k        = MINIBENCH_K_BASE
    triggers = ["T1(base)"]
    if agree:
        k = min(k + MINIBENCH_K_AGREE, MINIBENCH_K_MAX); triggers.append("T2(agree)")
    if strong:
        k = min(k + MINIBENCH_K_RESEARCH, MINIBENCH_K_MAX); triggers.append("T3(research)")

    result = clamp01(_sigmoid(_logit(clamp01(blend, 1e-6, 1 - 1e-6)) * k))

    if MINIBENCH_GATE_LO <= result <= MINIBENCH_GATE_HI:
        result = clamp01(_sigmoid(_logit(clamp01(result, 1e-6, 1 - 1e-6)) * MINIBENCH_GATE_AMP))
        triggers.append("T4(gate)")

    return result, k, "+".join(triggers)

# ============================================================
# TAVILY / ASKNEWS / YFINANCE HELPERS
# ============================================================
def build_asknews_query(question: MetaculusQuestion, max_chars: int = 397) -> str:
    q = (question.question_text or "").strip()
    bg = (question.background_info or "").strip()

    q = re.sub(r"http\S+", "", q)
    bg = re.sub(r"http\S+", "", bg)
    q = re.sub(r"\s+", " ", q).strip()
    bg = re.sub(r"\s+", " ", bg).strip()

    if len(q) <= max_chars:
        if not bg: return q
        candidate = f"{q} — {bg}"
        if len(candidate) <= max_chars: return candidate
        space_for_bg = max_chars - len(q) - 3
        if space_for_bg > 10:
            return f"{q} — {textwrap.shorten(bg, width=space_for_bg, placeholder='…')}"
        return q

    first_sent = q.split(".")[0].strip()
    if len(first_sent) > max_chars:
        return textwrap.shorten(first_sent, width=max_chars, placeholder="…")

    remaining = max_chars - len(first_sent) - 3
    if remaining > 10 and bg:
        combo = f"{first_sent} — {textwrap.shorten(bg, width=remaining, placeholder='…')}"
        if len(combo) <= max_chars: return combo

    return textwrap.shorten(q, width=max_chars, placeholder="…")

def _get_tavily_client() -> "TavilyClient":
    if not TAVILY_AVAILABLE:
        raise RuntimeError("Tavily is not installed. pip install tavily")
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is required.")
    return TavilyClient(api_key=api_key)

def _sync_tavily_search(query: str, max_results: int = 5) -> List[str]:
    client = _get_tavily_client()
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
            content = (res.get("content", "") or "")[:600]
            url = res.get("url", "")
            snippet = f"[T{i+1}] {title}: {textwrap.shorten(content, width=260, placeholder='…')}" + (f" ({url})" if url else "")
            snippets.append(snippet)
        return snippets
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return []

def _fetch_yfinance_data_sync(ticker: str) -> str:
    if not YFINANCE_AVAILABLE:
        return "[yfinance package not installed. Run: pip install yfinance]"
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="3mo")
        if hist.empty:
            return f"[No live data found for ticker: {ticker}]"
            
        spot = hist['Close'].iloc[-1]
        high_52 = tk.info.get('fiftyTwoWeekHigh', 'N/A')
        low_52 = tk.info.get('fiftyTwoWeekLow', 'N/A')
        returns = hist['Close'].pct_change().dropna()
        vol = returns.std() * math.sqrt(252) 
        
        # Calculate naive 1-month random walk baseline based on recent volatility
        monthly_vol = vol * math.sqrt(21/252)
        rw_p10 = spot * math.exp(-1.28 * monthly_vol)
        rw_p90 = spot * math.exp(1.28 * monthly_vol)
        
        return (f"--- LIVE MARKET DATA ({ticker}) ---\n"
                f"Spot Price (Current): {spot:.2f}\n"
                f"52-Week Range: {low_52} - {high_52}\n"
                f"Annualized Volatility (3mo): {vol:.2%}\n"
                f"Random Walk Baseline (1-Month out): P10={rw_p10:.2f}, P50={spot:.2f}, P90={rw_p90:.2f}\n"
                f"*(Note: Random walk ignores macro trends and assumes log-normal price movement)*\n")
    except Exception as e:
        logger.error(f"yfinance fetch error for {ticker}: {e}")
        return f"[yfinance fetch error for {ticker}]"

# ============================================================
# Logging + env
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("samcodes")

ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_CLIENT_SECRET = os.getenv("ASKNEWS_CLIENT_SECRET")
ASKNEWS_ENABLED = bool(ASKNEWS_CLIENT_ID and ASKNEWS_CLIENT_SECRET)

# ============================================================
# Bot Class
# ============================================================
class samcodes(ForecastBot):
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    def _llm_config_defaults(self) -> Dict[str, str]:
        defaults = super()._llm_config_defaults()
        defaults.update({
            "researcher": "openrouter/openai/gpt-5.4",
            "forecaster_gpt": "openrouter/openai/gpt-5.4",
            "forecaster_claude": "openrouter/anthropic/claude-sonnet-4.6",
            "parser": "openrouter/anthropic/claude-sonnet-4.5",
        })
        return defaults

    def __init__(
        self,
        *args,
        extremize_enabled: bool = False,
        extremize_k_binary: float = 1.0,
        extremize_k_mc: float = 1.0,
        fat_tails_enabled: bool = True,
        tail_fatten_factor: float = 1.15,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._asknews_client = None
        self.extremize_enabled = bool(extremize_enabled)
        self.extremize_k_binary = float(extremize_k_binary)
        self.extremize_k_mc = float(extremize_k_mc)
        
        # New Fat Tail parameters for Market Pulse
        self.fat_tails_enabled = bool(fat_tails_enabled)
        self.tail_fatten_factor = float(tail_fatten_factor)
        
        self._active_tournament: str = ""

        logger.info(
            f"Initialized samcodes (AskNews={ASKNEWS_ENABLED}, yfinance={YFINANCE_AVAILABLE}, "
            f"fat_tails={self.fat_tails_enabled} factor={self.tail_fatten_factor})"
        )

        self._drop_counts: Dict[str, int] = {}
        self._drop_counts_by_model: Dict[str, Dict[str, int]] = {"gpt": {}, "claude": {}}

    def set_active_tournament(self, tid: str) -> None:
        self._active_tournament = str(tid).strip().lower()
        logger.info(f"Active tournament: '{self._active_tournament}'")

    def _inc_drop(self, model_tag: str, reason: str) -> None:
        self._drop_counts[reason] = self._drop_counts.get(reason, 0) + 1
        d = self._drop_counts_by_model.get(model_tag, {})
        d[reason] = d.get(reason, 0) + 1
        self._drop_counts_by_model[model_tag] = d

    # -----------------------------
    # AskNews Client
    # -----------------------------
    def _get_asknews_client(self):
        if self._asknews_client is not None: return self._asknews_client
        if not ASKNEWS_ENABLED:
            self._asknews_client = None
            return None

        if ASKNEWS_SDK_AVAILABLE:
            self._asknews_client = AskNewsSDK(
                client_id=ASKNEWS_CLIENT_ID, client_secret=ASKNEWS_CLIENT_SECRET, scopes=["news"],
            )
        else:
            auth_url = "https://api.asknews.app/v1/oauth/token"
            data = {"grant_type": "client_credentials", "client_id": ASKNEWS_CLIENT_ID, "client_secret": ASKNEWS_CLIENT_SECRET, "scope": "news"}
            try:
                resp = requests.post(auth_url, data=data, timeout=10)
                resp.raise_for_status()
                self._asknews_client = {"token": resp.json()["access_token"]}
            except Exception as e:
                logger.error(f"AskNews Auth error: {e}")
                self._asknews_client = None
        return self._asknews_client

    def _sync_asknews_search(self, query: str) -> List[Any]:
        client = self._get_asknews_client()
        if client is None: return []

        if ASKNEWS_SDK_AVAILABLE:
            try:
                response = client.news.search_news(query=query, n_articles=5, return_type="news", method="kw", return_story_text=True)
                if hasattr(response, "news"): return response.news
                data = getattr(response, "data", response if isinstance(response, dict) else {})
                return data.get("news", []) if isinstance(data, dict) else []
            except Exception as e:
                logger.error(f"AskNews SDK search error: {e}")
                return []
        else:
            headers = {"Authorization": f"Bearer {client['token']}"}
            params = {"q": query, "n_articles": 5, "sort": "relevance", "return_type": "news", "use_neural_search": "false", "return_story_text": "true"}
            try:
                resp = requests.get("https://api.asknews.app/v1/news", headers=headers, params=params, timeout=15)
                resp.raise_for_status()
                return resp.json().get("data", resp.json()).get("news", [])
            except Exception as e:
                logger.error(f"AskNews HTTP search error: {e}")
                return []

    # -----------------------------
    # Research Pipeline + Ticker Extractor
    # -----------------------------
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            today_str = datetime.now().strftime("%Y-%m-%d")
            query = build_asknews_query(question)
            
            # ROUTE 1: Live yfinance data extraction
            financial_data = ""
            if "market-pulse" in self._active_tournament:
                extract_prompt = (
                    f"Extract the single most relevant Yahoo Finance ticker for this question. "
                    f"Reply ONLY with the valid ticker symbol (e.g. AAPL, ^GSPC, BTC-USD, CL=F). "
                    f"If it is a macroeconomic indicator without a direct ticker, reply NONE. "
                    f"Question: {question.question_text}"
                )
                try:
                    ticker = await self._invoke_llm("parser", extract_prompt)
                    ticker = ticker.strip().upper()
                    if ticker and ticker != "NONE":
                        loop = asyncio.get_running_loop()
                        financial_data = await loop.run_in_executor(None, _fetch_yfinance_data_sync, ticker) + "\n"
                except Exception as e:
                    logger.warning(f"Ticker extraction failed: {e}")

            # Standard AskNews + Tavily Search
            asknews_summary = "[AskNews disabled]"
            if ASKNEWS_ENABLED:
                try:
                    loop = asyncio.get_running_loop()
                    stories = await loop.run_in_executor(None, self._sync_asknews_search, query)
                    if stories:
                        snippets = []
                        for i, story in enumerate(stories[:5]):
                            title = story.get("title", "Untitled") if isinstance(story, dict) else getattr(story, "title", "Untitled")
                            text = (story.get("text") or "")[:700] if isinstance(story, dict) else (getattr(story, "text", "") or "")[:700]
                            snippets.append(f"[A{i+1}] {title}: {textwrap.shorten(text, width=260, placeholder='…')}")
                        asknews_summary = "\n".join(snippets)
                    else:
                        asknews_summary = "[AskNews: No recent stories found]"
                except Exception as e:
                    asknews_summary = f"[AskNews error: {str(e)}]"

            tavily_summary = "[Tavily: No results found]"
            try:
                loop = asyncio.get_running_loop()
                tavily_snippets = await loop.run_in_executor(None, _sync_tavily_search, query)
                tavily_summary = "\n".join(tavily_snippets) if tavily_snippets else "[Tavily: No results found]"
            except Exception as e:
                tavily_summary = f"[Tavily error: {str(e)}]"

            return (
                f"{financial_data}"
                f"--- LIVE NEWS (as of {today_str}) ---\n{asknews_summary}\n\n"
                f"--- WEB SEARCH SNIPPETS (TAVILY) ---\n{tavily_summary}\n"
            )

    # -----------------------------
    # Parsers
    # -----------------------------
    async def _invoke_llm(self, model_name: str, prompt: str) -> str:
        llm = self.get_llm(model_name, "llm")
        return await llm.invoke(prompt)

    async def _invoke_with_format_retry(self, model_name: str, prompt: str, format_spec: str) -> str:
        return await self._invoke_llm(model_name, prompt)

    async def _parse_binary(self, raw: str, model_tag: str) -> Optional[float]:
        try:
            pred: BinaryPrediction = await structure_output(
                text_to_structure=raw, output_type=BinaryPrediction, model=self.get_llm("parser", "llm"), num_validation_samples=self._structure_output_validation_samples,
            )
            val = safe_float(getattr(pred, "prediction_in_decimal", None), default=None)
            if val is not None: return clamp01(float(val))
        except Exception:
            self._inc_drop(model_tag, "parse_error_binary_structured")

        val2 = extract_binary_prob_from_text(raw)
        if val2 is None:
            self._inc_drop(model_tag, "parse_error_binary_fallback")
            return None
        return clamp01(float(val2))

    async def _parse_mc(self, raw: str, question: MultipleChoiceQuestion, model_tag: str) -> Optional[Dict[str, float]]:
        options = list(question.options)
        try:
            pred: PredictedOptionList = await structure_output(
                text_to_structure=raw, output_type=PredictedOptionList, model=self.get_llm("parser", "llm"),
                additional_instructions=f"Valid options: {options}. Prefer interpreting numbered options if present.",
                num_validation_samples=self._structure_output_validation_samples,
            )
            pred_dict = {str(po.option_name).strip(): float(po.probability) for po in pred.predicted_options if _is_num(po.probability)}
            out: Dict[str, float] = {}
            for opt in options:
                if opt in pred_dict: out[opt] = pred_dict[opt]
                else:
                    opt_cf = opt.casefold()
                    for k, v in pred_dict.items():
                        if k.casefold() == opt_cf:
                            out[opt] = v; break
            if out: return out
        except Exception:
            self._inc_drop(model_tag, "parse_error_mc_structured")

        idx_probs = extract_indexed_mc_probs(raw, len(options))
        if not idx_probs:
            self._inc_drop(model_tag, "parse_error_mc_fallback")
            return None
        out2 = {options[i - 1]: float(idx_probs[i]) for i in range(1, len(options) + 1) if i in idx_probs}
        if not out2:
            self._inc_drop(model_tag, "parse_error_mc_fallback_empty")
            return None
        return out2

    async def _parse_numeric(self, raw: str, question: NumericQuestion, model_tag: str) -> Optional[NumericDistribution]:
        targets = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        try:
            percentile_list: List[Percentile] = await structure_output(
                text_to_structure=raw, output_type=list[Percentile], model=self.get_llm("parser", "llm"), num_validation_samples=self._structure_output_validation_samples,
            )
            clean_percentiles: List[Percentile] = []
            for p in percentile_list:
                val = safe_float(getattr(p, "value", None), default=None)
                if val is not None:
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

        extracted = extract_numeric_percentiles(raw, targets)
        if not extracted:
            self._inc_drop(model_tag, "parse_error_numeric_fallback")
            return None

        pts = sorted([Percentile(percentile=pt, value=float(extracted[pt])) for pt in targets if pt in extracted], key=lambda x: float(x.percentile))
        for i in range(1, len(pts)):
            if pts[i].value < pts[i - 1].value: pts[i].value = pts[i - 1].value

        return NumericDistribution.from_question(pts, question)

    def _create_upper_and_lower_bound_messages(self, question: NumericQuestion) -> Tuple[str, str]:
        low = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
        high = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
        low_msg = f"Cannot be lower than {low}." if not question.open_lower_bound else f"Unlikely below {low}."
        high_msg = f"Cannot be higher than {high}." if not question.open_upper_bound else f"Unlikely above {high}."
        return low_msg, high_msg

    # -----------------------------
    # Prompts (Including ROUTE 2: Quant Prompt)
    # -----------------------------
    def _binary_prompt(self, question: BinaryQuestion, research: str, role: str) -> str:
        return clean_indents(f"""
            You are forecasting to maximize proper scoring rules (log score / Brier). Be calibrated and decisive.
            Role: {role}
            Today: {datetime.now().strftime('%Y-%m-%d')}

            Question: {question.question_text}
            Resolution Criteria: {question.resolution_criteria}

            Research:
            {research}

            Write 6-10 bullets total, then output EXACTLY these two lines at the end:
            Probability: ZZ%
            Decimal: 0.ZZ
        """)

    def _mc_prompt(self, question: MultipleChoiceQuestion, research: str, role: str) -> str:
        options = build_indexed_options(list(question.options))
        return clean_indents(f"""
            You are forecasting to maximize proper scoring rules. Be calibrated and decisive.
            Role: {role}
            Today: {datetime.now().strftime('%Y-%m-%d')}

            Question: {question.question_text}
            Resolution Criteria: {question.resolution_criteria}

            Options:
            {chr(10).join(options)}

            Research:
            {research}

            Write 6-10 bullets total, then output EXACTLY n lines at the end (summing to 100%):
            1: XX%
            2: XX%
            ...
        """)

    def _numeric_prompt(self, question: NumericQuestion, research: str, role: str) -> str:
        low_msg, high_msg = self._create_upper_and_lower_bound_messages(question)
        unit = getattr(question, "unit_of_measure", "inferred")
        
        # ROUTE 2: Quantitative Chain of Thought for Market Pulse
        if "market-pulse" in self._active_tournament:
            prompt_body = f"""
            You are a Quantitative Analyst modeling financial and macroeconomic distributions.
            Your goal is to build a highly calibrated probabilistic model avoiding overconfidence.

            Question: {question.question_text}
            Resolution Criteria: {question.resolution_criteria}
            Units: {unit} | Bounds: {low_msg} {high_msg}

            Market & Research Data:
            {research}

            Perform this specific chain of thought:
            1. Identify the current EXACT Spot Price/Value of the asset or metric.
            2. Identify the target date and calculate the required delta to hit historical averages.
            3. Assess Historical Volatility: How often has the asset swung this wildly in this timeframe?
            4. If provided, anchor heavily to the Random Walk Baseline, adjusting only for specific macro headwinds/tailwinds.
            5. Beware of narrow tails: Financial markets have fat tails. Push your P10 and P90 wider to account for shocks.

            Write out your math, then end EXACTLY with these lines:
            Percentile 10: X
            Percentile 20: X
            Percentile 40: X
            Percentile 60: X
            Percentile 80: X
            Percentile 90: X
            """
        else:
            prompt_body = f"""
            You are forecasting to maximize proper scoring rules. Be calibrated and decisive.
            Apply outside-view base rates first.
            
            Question: {question.question_text}
            Resolution Criteria: {question.resolution_criteria}
            Units: {unit} | Bounds: {low_msg} {high_msg}

            Research:
            {research}

            Write 6-10 bullets, then end EXACTLY with these lines:
            Percentile 10: X
            ...
            Percentile 90: X
            """
        
        return clean_indents(f"Role: {role}\nToday: {datetime.now().strftime('%Y-%m-%d')}\n{prompt_body}")

    # -----------------------------
    # Role runs
    # -----------------------------
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
            try:
                raw2 = await self._invoke_llm(model_name, "Output ONLY: Probability: ZZ%\nDecimal: 0.ZZ")
                val = await self._parse_binary(raw2, model_tag=model_tag)
                raw += "\n\n[FORMAT_RETRY]\n" + raw2
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
            try:
                raw2 = await self._invoke_llm(model_name, "Output ONLY numbered percentage lines summing to 100%.")
                probs = await self._parse_mc(raw2, question, model_tag=model_tag)
                raw += "\n\n[FORMAT_RETRY]\n" + raw2
            except Exception:
                self._inc_drop(model_tag, "retry_failed_mc")

        if probs is None:
            self._inc_drop(model_tag, "invalid_values_mc")
            options = list(question.options)
            probs = {opt: 1.0 / max(1, len(options)) for opt in options}

        predicted_options_list = [PredictedOption(option_name=opt, probability=float(p)) for opt, p in probs.items()]
        return ReasonedPrediction(prediction_value=PredictedOptionList(predicted_options=predicted_options_list), reasoning=raw)

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
            try:
                raw2 = await self._invoke_llm(model_name, "Output ONLY Percentile 10 to 90 lines.")
                dist = await self._parse_numeric(raw2, question, model_tag=model_tag)
                raw += "\n\n[FORMAT_RETRY]\n" + raw2
            except Exception:
                self._inc_drop(model_tag, "retry_failed_numeric")

        if dist is None:
            self._inc_drop(model_tag, "invalid_values_numeric")
            l = float(question.lower_bound) if question.lower_bound is not None else 0.0
            u = float(question.upper_bound) if question.upper_bound is not None else 100.0
            dist = NumericDistribution.from_question([Percentile(value=(l+u)/2.0, percentile=0.5)], question)

        return ReasonedPrediction(prediction_value=dist, reasoning=raw)

    async def _run_forecast_on_binary(self, q, r): return await self._run_binary_role(q, r, "gpt", "PRIMARY")
    async def _run_forecast_on_multiple_choice(self, q, r): return await self._run_mc_role(q, r, "gpt", "PRIMARY")
    async def _run_forecast_on_numeric(self, q, r): return await self._run_numeric_role(q, r, "gpt", "PRIMARY")

    # -----------------------------
    # Aggregation & EXTREMIZATION
    # -----------------------------
    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        async with self._concurrency_limiter:
            preds: List[Any] = []
            reasonings: List[str] = []
            is_mb = self._active_tournament == "minibench"

            try:
                if isinstance(question, BinaryQuestion): gpt_pred = await self._run_binary_role(question, research, "gpt", "PRIMARY")
                elif isinstance(question, MultipleChoiceQuestion): gpt_pred = await self._run_mc_role(question, research, "gpt", "PRIMARY")
                else: gpt_pred = await self._run_numeric_role(question, research, "gpt", "PRIMARY")
                preds.append(gpt_pred.prediction_value)
                reasonings.append("[GPT_PRIMARY]\n" + gpt_pred.reasoning)
            except Exception as e: logger.error(f"GPT primary failed: {e}")

            try:
                if isinstance(question, BinaryQuestion): cl_pred = await self._run_binary_role(question, research, "claude", "CHECKER")
                elif isinstance(question, MultipleChoiceQuestion): cl_pred = await self._run_mc_role(question, research, "claude", "CHECKER")
                else: cl_pred = await self._run_numeric_role(question, research, "claude", "CHECKER")
                preds.append(cl_pred.prediction_value)
                reasonings.append("[CLAUDE_CHECKER]\n" + cl_pred.reasoning)
            except Exception as e: logger.error(f"Claude checker failed: {e}")

            if not preds: raise RuntimeError("All forecasters failed.")
            w_gpt, w_claude = 0.70, 0.30

            # --- BINARY ---
            if isinstance(question, BinaryQuestion):
                g = float(preds[0]) if len(preds) >= 1 and _is_num(preds[0]) else None
                c = float(preds[1]) if len(preds) >= 2 and _is_num(preds[1]) else None

                blend = clamp01(w_gpt * g + w_claude * c) if (g and c) else clamp01(g or c or 0.5)
                numeric_preds = [v for v in (g, c) if v is not None] or [0.5]

                final, ext_log = blend, "extremize=OFF"
                if self.extremize_enabled:
                    if is_mb:
                        final, eff_k, trigs = minibench_extremize_binary(blend, g, c, research)
                        ext_log = f"minibench({trigs} k_eff={eff_k:.1f}) {blend:.3f}->{final:.3f}"
                    elif self.extremize_k_binary and abs(self.extremize_k_binary - 1.0) > 1e-12:
                        final = extremize_binary(blend, self.extremize_k_binary)
                        ext_log = f"extremize_bin(k={self.extremize_k_binary:.3f}) {blend:.3f}->{final:.3f}"

                stats_line = f"[stats] n={len(numeric_preds)} mean={mean(numeric_preds):.3f} agg=weighted(0.7/0.3) {ext_log}"
                return ReasonedPrediction(prediction_value=final, reasoning=stats_line + "\n\n" + "\n\n---\n\n".join(reasonings))

            # --- MULTIPLE CHOICE ---
            if isinstance(question, MultipleChoiceQuestion):
                options = list(question.options)
                g_dict = {str(po.option_name).strip(): float(po.probability) for po in preds[0].predicted_options if _is_num(po.probability)} if len(preds) >= 1 and isinstance(preds[0], PredictedOptionList) else {}
                c_dict = {str(po.option_name).strip(): float(po.probability) for po in preds[1].predicted_options if _is_num(po.probability)} if len(preds) >= 2 and isinstance(preds[1], PredictedOptionList) else {}

                blended = {}
                for opt in options:
                    gv, cv = g_dict.get(opt), c_dict.get(opt)
                    if gv is None and cv is None: blended[opt] = 1e-6
                    elif gv is None: blended[opt] = float(cv)
                    elif cv is None: blended[opt] = float(gv)
                    else: blended[opt] = w_gpt * float(gv) + w_claude * float(cv)

                total = sum(blended.values())
                blended = {k: v / total for k, v in blended.items()} if total > 0 else {opt: 1.0/max(1, len(options)) for opt in options}

                ext_log = "extremize=OFF"
                if self.extremize_enabled:
                    k_mc = MINIBENCH_K_MC if is_mb else self.extremize_k_mc
                    if abs(k_mc - 1.0) > 1e-12:
                        blended = extremize_mc(blended, k_mc)
                        ext_log = f"extremize_mc(k={k_mc:.3f}{'[mb]' if is_mb else ''})"

                stats_line = f"[stats] n={len(preds)} entropy={entropy(blended):.3f} agg=weighted(0.7/0.3) {ext_log}"
                predicted_options_list = [PredictedOption(option_name=opt, probability=float(prob)) for opt, prob in blended.items()]
                return ReasonedPrediction(prediction_value=PredictedOptionList(predicted_options=predicted_options_list), reasoning=stats_line + "\n\n" + "\n\n---\n\n".join(reasonings))

            # --- NUMERIC (ROUTE 3: Tail Fattening) ---
            if isinstance(question, NumericQuestion):
                targets = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
                
                def dist_to_map(d: Any) -> Dict[float, float]:
                    return {normalize_percentile(getattr(item, "percentile", None)): float(safe_float(getattr(item, "value", None)))
                            for item in getattr(d, "declared_percentiles", []) if safe_float(getattr(item, "value", None)) is not None}

                g_map = dist_to_map(preds[0]) if len(preds) >= 1 else {}
                c_map = dist_to_map(preds[1]) if len(preds) >= 2 else {}

                blended_pts: List[Percentile] = []
                for pt in targets:
                    gv = min(g_map.items(), key=lambda kv: abs(kv[0] - pt))[1] if g_map else None
                    cv = min(c_map.items(), key=lambda kv: abs(kv[0] - pt))[1] if c_map else None

                    if gv is None and cv is None:
                        v = (float(getattr(question, "lower_bound", 0.0) or 0.0) + float(getattr(question, "upper_bound", 100.0) or 100.0)) / 2.0
                    elif gv is None: v = float(cv)
                    elif cv is None: v = float(gv)
                    else: v = w_gpt * float(gv) + w_claude * float(cv)

                    blended_pts.append(Percentile(percentile=pt, value=float(v)))

                blended_pts.sort(key=lambda x: float(x.percentile))
                for i in range(1, len(blended_pts)):
                    if blended_pts[i].value < blended_pts[i - 1].value: blended_pts[i].value = blended_pts[i - 1].value

                # Apply Fat Tails for Market Pulse
                ext_log = "fat_tails=OFF"
                if self.fat_tails_enabled and "market-pulse" in self._active_tournament:
                    blended_pts = apply_tail_fattening(blended_pts, self.tail_fatten_factor)
                    ext_log = f"tail_fatten(factor={self.tail_fatten_factor:.2f})"

                p10 = next((p.value for p in blended_pts if abs(float(p.percentile) - 0.1) < 1e-9), None)
                p90 = next((p.value for p in blended_pts if abs(float(p.percentile) - 0.9) < 1e-9), None)
                spread = (p90 - p10) if (p10 is not None and p90 is not None) else float("nan")
                
                stats_line = f"[stats] n={len(preds)} p10={float(p10 or 0):.3f} p90={float(p90 or 0):.3f} spread={float(spread):.3f} agg=weighted(0.7/0.3) {ext_log}"

                final_dist = NumericDistribution.from_question(blended_pts, question)
                return ReasonedPrediction(prediction_value=final_dist, reasoning=stats_line + "\n\n" + "\n\n---\n\n".join(reasonings))

            return ReasonedPrediction(prediction_value=preds[0], reasoning="\n\n---\n\n".join(reasonings))

    def log_internal_drop_stats(self) -> None:
        if not self._drop_counts: return
        logger.info(f"[drops] totals={self._drop_counts}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    litellm_logger = logging.getLogger("litellm")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(description="Run samcodes with yfinance + AskNews + Extremization")
    parser.add_argument("--tournament-ids", nargs="+", type=str, default=["minibench", "32916", "market-pulse-26q2"])
    parser.add_argument("--extremize", action="store_true", help="Enable extremization for Binary + MC probabilities")
    parser.add_argument("--extremize-k-binary", type=float, default=1.0)
    parser.add_argument("--extremize-k-mc", type=float, default=1.0)
    parser.add_argument("--fat-tails", action="store_true", default=True, help="Widen P10/P90 for numeric questions")
    parser.add_argument("--tail-fatten-factor", type=float, default=1.15)

    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"): raise SystemExit("❌ OPENROUTER_API_KEY is required")
    if not TAVILY_AVAILABLE or not os.getenv("TAVILY_API_KEY"): raise SystemExit("❌ Tavily package/key required")

    bot = samcodes(
        research_reports_per_question=1,
        predictions_per_research_report=2,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        extremize_enabled=bool(args.extremize),
        extremize_k_binary=float(args.extremize_k_binary),
        extremize_k_mc=float(args.extremize_k_mc),
        fat_tails_enabled=bool(args.fat_tails),
        tail_fatten_factor=float(args.tail_fatten_factor),
    )

    async def run_all():
        all_reports = []
        for tid in args.tournament_ids:
            bot.set_active_tournament(tid)
            logger.info(f"▶️ Forecasting on tournament: {tid}")
            reports = await bot.forecast_on_tournament(tid, return_exceptions=True)
            all_reports.extend(reports)
        return all_reports

    try:
        reports = asyncio.run(run_all())
        bot.log_report_summary(reports)
        bot.log_internal_drop_stats()
        logger.info("✅ samcodes run completed.")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise SystemExit(1)
