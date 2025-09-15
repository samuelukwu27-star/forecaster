# ./bots/top_model_bot.py
# Contains the core logic for the "Top Model Bot".

import os
import requests
import re
import datetime
from bs4 import BeautifulSoup

class TopModelBot:
    """
    Implements the "Top Model" strategy: uses a powerful free model via OpenRouter
    with a direct, robust prompt inspired by the metac-o1 bot.
    """
    def __init__(self):
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        # Using a fast, free, and capable model available on OpenRouter
        self.model = "google/gemini-flash-1.5"
        self.openrouter_api_url = "https://openrouter.ai/api/v1/chat/completions"
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

    def _get_web_scrape_summary(self, query):
        """Performs a web search and scrapes the top result for a summary."""
        print(f"Scraping web for: '{query}'")
        try:
            # Using DuckDuckGo's HTML version for simple scraping.
            search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            
            search_response = requests.get(search_url, headers=headers, timeout=15)
            search_response.raise_for_status()

            soup = BeautifulSoup(search_response.text, 'html.parser')
            first_link = soup.find('a', class_='result__a')
            
            if not first_link or not first_link.get('href'):
                return "Web scraping found no relevant links."

            article_url = first_link['href']
            print(f"Scraping content from: {article_url}")
            
            article_response = requests.get(article_url, headers=headers, timeout=15)
            article_response.raise_for_status()

            article_soup = BeautifulSoup(article_response.text, 'html.parser')
            paragraphs = article_soup.find_all('p', limit=5)
            scraped_text = ' '.join([p.get_text() for p in paragraphs if p.get_text()])

            if not scraped_text:
                return "Could not extract meaningful text from the top search result."

            summary = (scraped_text[:800] + '...') if len(scraped_text) > 800 else scraped_text
            return f"Summary Report: {summary}"

        except requests.RequestException as e:
            print(f"Web scraping failed: {e}")
            return "Summary Report: Web search failed. Proceeding with no external data."

    def _get_binary_question_prompt(self, question):
        """Generates the prompt for the LLM, based on the successful template."""
        today = datetime.date.today().strftime("%B %d, %Y")
        summary = self._get_web_scrape_summary(question['title'])
        return f"""
You are a professional forecaster. Your interview question is:
{question['title']}
Background: {question.get('background', '')}
Resolution Criteria: {question.get('resolution_criteria', '')}

Your research assistant says:
{summary}

Today is {today}.
Before answering, you will write:
(a) Time left until resolution.
(b) The status quo outcome.
(c) A scenario for a 'No' outcome.
(d) A scenario for a 'Yes' outcome.
Write your rationale, remembering to weigh the status quo heavily.
Finally, write your answer as: "Probability: ZZ%", 0-100
"""

    def _call_llm(self, prompt):
        """Calls the OpenRouter model and returns the content."""
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo/ai-forecasting-bots",
            "X-Title": "AI Forecasting Bot - Top Model"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        try:
            response = requests.post(self.openrouter_api_url, headers=headers, json=data, timeout=180)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.RequestException as e:
            print(f"Error calling LLM via OpenRouter: {e}")
            return None

    def forecast_question(self, question):
        """
        Takes a question dictionary, performs a forecast, and returns the result.
        """
        prompt = self._get_binary_question_prompt(question)
        content = self._call_llm(prompt)
        
        if not content:
            return None, "LLM call failed."

        match = re.search(r"Probability:\s*(\d{1,3})\s*%", content)
        if match:
            probability = float(match.group(1)) / 100.0
            return probability, content  # Return full content as rationale
        
        print("Warning: Could not parse probability from LLM response.")
        return None, content

