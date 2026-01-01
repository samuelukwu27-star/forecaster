import os
import json
import requests
import re
import time
import statistics
from openai import OpenAI
from tavily import TavilyClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 🎯 TARGETS LIST
# Strings = Tournament Slugs (e.g., "ACX2026")
# Integers = Specific Question IDs OR Project IDs (Script now detects which is which)
TARGETS = [
    "ACX2026",
    "market-pulse-26q1",
    32916  # This is a Project ID (Spring AIB 2026), script will now handle it correctly.
]

# ⚡ THE COUNCIL
FREE_MODELS = [
    "deepseek/deepseek-r1:free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "tngtech/deepseek-r1t2-chimera:free",
]

if not all([METACULUS_TOKEN, OPENROUTER_API_KEY, TAVILY_API_KEY]):
    raise ValueError("❌ Missing API Keys!")

tavily = TavilyClient(api_key=TAVILY_API_KEY)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={"HTTP-Referer": "https://github.com/bot", "X-Title": "Geobot Forecaster"}
)

# --- UTILS ---
def repair_json(text):
    """Attempts to extract and parse JSON even if it's messy."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Extract strictly between first { and last }
    start = text.find('{')
    end = text.rfind('}')
    
    if start == -1 or end == -1:
        return None
        
    json_str = text[start:end+1]
    
    try:
        return json.loads(json_str)
    except:
        # Naive cleanup for common trailing comma errors
        try:
            json_str = re.sub(r',\s*}', '}', json_str)
            return json.loads(json_str)
        except:
            return None

def get_headers():
    return {"Authorization": f"Token {METACULUS_TOKEN}"}

# --- 🚀 ROBUST FETCHING LOGIC ---
def fetch_questions_for_target(target):
    base_url = "https://www.metaculus.com/api2"
    
    # A. If Target is an Integer (Could be Question ID OR Project ID)
    if isinstance(target, int):
        print(f"🔍 Checking if {target} is a Question ID...")
        try:
            # 1. Try as Question
            resp = requests.get(f"{base_url}/questions/{target}/", headers=get_headers())
            if resp.status_code == 200:
                q = resp.json()
                if q['status'] == 'open':
                    return [q]
                else:
                    print(f"   ⚠️ Question {target} is closed.")
                    return []
            elif resp.status_code == 404:
                print(f"   ⚠️ {target} is not a Question ID. Checking Project ID...")
                # 2. Fallback: Treat as Project ID (e.g. 32916)
                return _fetch_from_tournament_or_project(target)
                
        except Exception as e:
            print(f"   ❌ Error checking ID {target}: {e}")
            return []

    # B. If Target is a String (Tournament Slug)
    return _fetch_from_tournament_or_project(target)

def _fetch_from_tournament_or_project(target_identifier):
    """Resolves a slug or ID to a list of questions."""
    base_url = "https://www.metaculus.com/api2"
    print(f"🔍 Resolving Container: '{target_identifier}'...")
    
    tournament_id = None
    
    # 1. If it's already an INT, it's the ID.
    if isinstance(target_identifier, int):
        tournament_id = target_identifier
        print(f"   ✅ Using ID: {tournament_id}")
    else:
        # 2. Resolve Slug -> ID
        try:
            # Try as Tournament
            resp = requests.get(f"{base_url}/tournaments/{target_identifier}/", headers=get_headers())
            if resp.status_code == 200:
                tournament_id = resp.json()['id']
                print(f"   ✅ Found Tournament ID: {tournament_id}")
            else:
                # Try as Project
                resp = requests.get(f"{base_url}/projects/{target_identifier}/", headers=get_headers())
                if resp.status_code == 200:
                    tournament_id = resp.json()['id']
                    print(f"   ✅ Found Project ID: {tournament_id}")
        except Exception as e:
            print(f"   ❌ Resolution failed: {e}")
            return []

    if not tournament_id:
        print(f"   ❌ Could not resolve '{target_identifier}'. Skipping.")
        return []

    # 3. Fetch Questions
    # Try 'tournament' param first, then 'project'
    params = {"status": "open", "type": "forecast", "limit": 10, "tournament": tournament_id}
    questions = _execute_list_query(params)
    
    if not questions:
        # Switch to project param
        del params['tournament']
        params['project'] = tournament_id
        questions = _execute_list_query(params)
        
    return questions

def _execute_list_query(params):
    time.sleep(1)
    try:
        resp = requests.get("https://www.metaculus.com/api2/questions/", params=params, headers=get_headers())
        if resp.status_code == 200:
            return resp.json()['results']
        return []
    except:
        return []

# --- 🧠 PROMPT FACTORY ---
def generate_prompt(q_data, evidence):
    title = q_data['title']
    background = q_data.get('description', '')[:1500]
    
    persona = "You are a Superforecaster. Analyze Evidence. Think in Base Rates. Return JSON ONLY."
    
    if q_data['type'] == 'forecast':
        return f"""
        {persona}
        QUESTION: {title}
        BACKGROUND: {background}
        NEWS: {evidence}
        TASK: JSON with probability (0-100) and concise rationale.
        OUTPUT: {{ "prediction": 65, "comment": "Base rate is..." }}
        """
    elif q_data['type'] in ['date', 'numeric']:
        return f"""
        {persona}
        QUESTION: {title}
        BACKGROUND: {background}
        NEWS: {evidence}
        TASK: JSON with p25, p50, p75.
        OUTPUT: {{ "prediction": {{ "p25": 10, "p50": 50, "p75": 90 }}, "comment": "Range based on..." }}
        """
    elif q_data['type'] == 'multiple_choice':
        options = q_data.get('possibilities', {}).get('format', {}).get('options', [])
        labels = [o if isinstance(o, str) else o.get('label') for o in options]
        return f"""
        {persona}
        QUESTION: {title}
        OPTIONS: {labels}
        NEWS: {evidence}
        TASK: JSON assigning % to options (Sum 100).
        OUTPUT: {{ "prediction": {{ "{labels[0]}": 20, "{labels[1] if len(labels)>1 else 'B'}": 80 }}, "comment": "Rationale..." }}
        """
    return None

# --- AGGREGATION ---
def aggregate_council_results(results, q_type):
    if not results: return None
    
    preds = [r['prediction'] for r in results]
    comments = [r['comment'] for r in results]
    best_comment = max(comments, key=len) if comments else "No comment."
    
    final_pred = None
    
    if q_type == 'forecast':
        valid = [p for p in preds if isinstance(p, (int, float))]
        if valid: final_pred = statistics.median(valid)

    elif q_type in ['date', 'numeric']:
        p25s, p50s, p75s = [], [], []
        for p in preds:
            if isinstance(p, dict):
                p25s.append(p.get('p25', 0))
                p50s.append(p.get('p50', 0))
                p75s.append(p.get('p75', 0))
        if p50s:
            final_pred = {
                "p25": statistics.mean(p25s),
                "p50": statistics.mean(p50s),
                "p75": statistics.mean(p75s)
            }

    elif q_type == 'multiple_choice':
        agg = {}
        count = 0
        for p in preds:
            if isinstance(p, dict):
                count += 1
                for k, v in p.items():
                    agg[k] = agg.get(k, 0) + float(v)
        if count > 0:
            final_pred = {k: v/count for k, v in agg.items()}

    if not final_pred: return None
    return {"prediction": final_pred, "comment": best_comment}

# --- MAIN ---
def run_council(q_data):
    try:
        search = tavily.search(query=q_data['title'], max_results=3)
        evidence = "\n".join([f"- {r['content']}" for r in search['results']])
    except:
        evidence = "No recent news found."

    prompt = generate_prompt(q_data, evidence)
    if not prompt: return None

    print(f"   🗳️  Polling Council ({len(FREE_MODELS)} models)...")
    results = []
    
    for model in FREE_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=800
            )
            data = repair_json(resp.choices[0].message.content)
            if data and 'prediction' in data: results.append(data)
        except: pass
            
    return aggregate_council_results(results, q_data['type'])

def submit(q, result):
    print(f"      🚀 [WOULD SUBMIT]: {result['prediction']}")
    print(f"      💬 [WOULD COMMENT]: {result['comment'][:100]}...")
    # Uncomment to enable real submission
    # try:
    #     pid = q['id']
    #     val = result['prediction']
    #     if q['type'] == 'forecast': val = float(val)/100.0
    #     elif q['type'] == 'multiple_choice': val = [val.get(o['label'], 0) for o in q['possibilities']['format']['options']]
    #     requests.post(f"https://www.metaculus.com/api2/questions/{pid}/predict/", json={"prediction": val}, headers=get_headers())
    # except Exception as e: print(e)

def main():
    for target in TARGETS:
        print(f"\n🌍 Processing: {target}")
        questions = fetch_questions_for_target(target)
        
        if not questions:
            print("   ⚠️ No questions found.")
            continue
            
        print(f"   ✅ Found {len(questions)} questions.")
        for q in questions:
            print(f"\n   🔮 {q['title'][:50]}...")
            res = run_council(q)
            if res:
                submit(q, res)
            else:
                print("      ❌ Council Consensus Failed.")
            time.sleep(2)

if __name__ == "__main__":
    main()
