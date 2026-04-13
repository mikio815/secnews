#!/usr/bin/env python3
"""
Security News Fetcher
RSSフィードからセキュリティニュースを取得し、Gemini APIで翻訳・要約してJSONを生成する
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import feedparser
import google.generativeai as genai
import yaml

# プロジェクトルート
ROOT = Path(__file__).parent.parent
SOURCES_FILE = ROOT / "sources.yml"
OUTPUT_JSON = ROOT / "docs" / "news.json"
OUTPUT_HTML = ROOT / "docs" / "index.html"
TEMPLATE_FILE = ROOT / "scripts" / "template.html"

JST = ZoneInfo("Asia/Tokyo")
MAX_ARTICLES_PER_SOURCE = 5
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


def load_sources() -> list[dict]:
    with open(SOURCES_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [s for s in data["sources"] if s.get("enabled", True)]


def fetch_feed(source: dict) -> list[dict]:
    print(f"  取得中: {source['name']} ...")
    try:
        feed = feedparser.parse(source["url"])
        articles = []
        for entry in feed.entries[:MAX_ARTICLES_PER_SOURCE]:
            summary_raw = ""
            if hasattr(entry, "summary"):
                summary_raw = entry.summary
            elif hasattr(entry, "description"):
                summary_raw = entry.description

            # HTMLタグを簡易除去
            import re
            summary_raw = re.sub(r"<[^>]+>", "", summary_raw).strip()
            summary_raw = summary_raw[:500] if len(summary_raw) > 500 else summary_raw

            pub_date = ""
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                pub_date = dt.astimezone(JST).strftime("%Y-%m-%d %H:%M JST")

            articles.append({
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "summary_en": summary_raw,
                "published": pub_date,
                "source": source["name"],
                "category": source["category"],
                "title_ja": "",
                "summary_ja": "",
            })
        return articles
    except Exception as e:
        print(f"  エラー ({source['name']}): {e}", file=sys.stderr)
        return []


def translate_and_summarize(articles: list[dict]) -> list[dict]:
    if not GEMINI_API_KEY:
        print("警告: GEMINI_API_KEY が未設定です。翻訳・要約をスキップします。")
        for a in articles:
            a["title_ja"] = a["title"]
            a["summary_ja"] = a["summary_en"]
        return articles

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    for i, article in enumerate(articles):
        if not article["title"] and not article["summary_en"]:
            continue
        try:
            prompt = f"""以下のセキュリティニュースの記事タイトルと概要を日本語に翻訳・要約してください。
出力はJSON形式で返してください。

タイトル: {article['title']}
概要: {article['summary_en']}

出力形式:
{{
  "title_ja": "日本語タイトル",
  "summary_ja": "日本語で2〜3文の要約"
}}"""

            response = model.generate_content(prompt)
            text = response.text.strip()

            # JSON部分を抽出
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                article["title_ja"] = result.get("title_ja", article["title"])
                article["summary_ja"] = result.get("summary_ja", article["summary_en"])
            else:
                article["title_ja"] = article["title"]
                article["summary_ja"] = article["summary_en"]

            print(f"  翻訳完了 ({i+1}/{len(articles)}): {article['title'][:40]}...")
            time.sleep(1)  # レートリミット対策

        except Exception as e:
            print(f"  翻訳エラー: {e}", file=sys.stderr)
            article["title_ja"] = article["title"]
            article["summary_ja"] = article["summary_en"]

    return articles


def save_json(articles: list[dict]) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "updated": datetime.now(JST).strftime("%Y-%m-%d %H:%M JST"),
        "count": len(articles),
        "articles": articles,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"JSON保存: {OUTPUT_JSON} ({len(articles)}件)")


def generate_html(articles: list[dict], updated: str) -> None:
    with open(TEMPLATE_FILE, encoding="utf-8") as f:
        template = f.read()

    html = template.replace("{{UPDATED}}", updated)

    # カテゴリラベル
    category_labels = {
        "general": "総合",
        "malware": "マルウェア",
        "technical": "技術",
        "cve": "CVE",
    }
    category_colors = {
        "general": "#e74c3c",
        "malware": "#e67e22",
        "technical": "#2980b9",
        "cve": "#8e44ad",
    }

    cards_html = ""
    for article in articles:
        cat = article.get("category", "general")
        label = category_labels.get(cat, cat)
        color = category_colors.get(cat, "#666")
        title_ja = article["title_ja"] or article["title"]
        summary_ja = article["summary_ja"] or article["summary_en"]
        pub = article.get("published", "")
        source = article.get("source", "")
        url = article.get("url", "#")

        cards_html += f"""
        <article class="news-card" data-category="{cat}">
          <div class="card-meta">
            <span class="category-badge" style="background:{color}">{label}</span>
            <span class="source">{source}</span>
            <span class="date">{pub}</span>
          </div>
          <h2 class="card-title">
            <a href="{url}" target="_blank" rel="noopener">{title_ja}</a>
          </h2>
          <p class="card-summary">{summary_ja}</p>
        </article>
"""

    html = html.replace("{{ARTICLES}}", cards_html)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML保存: {OUTPUT_HTML}")


def main():
    print("=== Security News Fetcher ===")
    print(f"開始: {datetime.now(JST).strftime('%Y-%m-%d %H:%M JST')}")

    sources = load_sources()
    print(f"\nソース数: {len(sources)}")

    all_articles = []
    for source in sources:
        articles = fetch_feed(source)
        all_articles.extend(articles)
        time.sleep(0.5)

    print(f"\n取得記事数: {len(all_articles)}")
    print("\n翻訳・要約中...")
    all_articles = translate_and_summarize(all_articles)

    updated = datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")
    save_json(all_articles)
    generate_html(all_articles, updated)

    print("\n完了.")


if __name__ == "__main__":
    main()
