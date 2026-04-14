#!/usr/bin/env python3
"""
Security News Fetcher
RSSフィードからセキュリティニュースを取得し、Gemini APIで翻訳・要約してJSONを生成する
"""

import html as html_mod
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

import feedparser
from groq import Groq
import yaml

# プロジェクトルート
ROOT = Path(__file__).parent.parent
SOURCES_FILE = ROOT / "sources.yml"
OUTPUT_JSON = ROOT / "docs" / "news.json"
OUTPUT_HTML = ROOT / "docs" / "index.html"
TEMPLATE_FILE = ROOT / "scripts" / "template.html"

JST = ZoneInfo("Asia/Tokyo")
MAX_ARTICLES_PER_SOURCE = 8
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


class _TextExtractor(HTMLParser):
    """HTMLからプレーンテキストを抽出するパーサー"""
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def strip_html(text: str) -> str:
    """HTMLタグとエンティティを除去してプレーンテキストを返す。
    正規表現ではなくHTMLParserを使うことで、属性値内の > など
    正規表現が誤検知するケースを防ぐ。"""
    extractor = _TextExtractor()
    try:
        extractor.feed(text)
        return extractor.get_text().strip()
    except Exception:
        # パース失敗時は正規表現でフォールバック
        return re.sub(r"<[^>]+>", "", text).strip()


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

            # HTMLタグとエンティティを除去してプレーンテキストにする
            summary_raw = strip_html(summary_raw)
            summary_raw = summary_raw[:500] if len(summary_raw) > 500 else summary_raw

            pub_date = ""
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                pub_date = dt.astimezone(JST).strftime("%Y-%m-%d %H:%M JST")

            articles.append({
                "title": strip_html(entry.get("title", "")),
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


BATCH_SIZE = 5   # 1リクエストで処理する記事数
BATCH_INTERVAL = 20  # TPM=6,000制限対策（1バッチ≈1,750トークン、3バッチ/分が上限）
MAX_RETRIES = 2
GROQ_MODEL = "llama-3.3-70b-versatile"


def _generate_with_retry(client: Groq, prompt: str) -> str:
    """429時にエラーメッセージの待機時間に従ってリトライする"""
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < MAX_RETRIES:
                m = re.search(r"retry after (\d+\.?\d*)s", err, re.IGNORECASE)
                wait = float(m.group(1)) + 5 if m else 30
                print(f"  レートリミット超過、{wait:.0f}秒待機してリトライ ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    return ""


def translate_and_summarize(articles: list[dict]) -> list[dict]:
    if not GROQ_API_KEY:
        print("警告: GROQ_API_KEY が未設定です。翻訳をスキップします。")
        for a in articles:
            a["title_ja"] = a["title"]
            a["summary_ja"] = a["summary_en"]
        return articles

    client = Groq(api_key=GROQ_API_KEY)

    # バッチ処理: BATCH_SIZE件ずつまとめて1リクエストで翻訳
    for batch_start in range(0, len(articles), BATCH_SIZE):
        batch = articles[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(articles) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  翻訳バッチ {batch_num}/{total_batches} ({len(batch)}件)...")

        # バッチをJSON配列としてプロンプトに渡す
        articles_json = json.dumps(
            [{"id": i, "title": a["title"], "summary": a["summary_en"]} for i, a in enumerate(batch)],
            ensure_ascii=False,
        )

        prompt = f"""あなたはサイバーセキュリティの専門家です。以下のセキュリティニュース記事リストを日本語に翻訳・要約してください。

ルール:
- 各記事の "id" はそのまま保持する
- "title_ja": タイトルを自然な日本語に翻訳する
- "summary_ja": 2〜3文の日本語要約
- exploit, ransomware, malware, zero-day, CVE, phishing など定着しているセキュリティ用語はカタカナ表記を優先する
- privilege escalation→権限昇格, vulnerability→脆弱性, threat actor→脅威アクター など文脈に合った訳語を使う
- JSON配列のみを返し、余計な説明は不要

{articles_json}

出力形式:
[
  {{"id": 0, "title_ja": "...", "summary_ja": "..."}},
  ...
]"""

        try:
            text = _generate_with_retry(client, prompt)

            # JSON配列部分を抽出
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                results = json.loads(match.group())
                result_map = {r["id"]: r for r in results if "id" in r}
                for i, article in enumerate(batch):
                    r = result_map.get(i, {})
                    article["title_ja"] = r.get("title_ja", article["title"])
                    article["summary_ja"] = r.get("summary_ja", article["summary_en"])
            else:
                for article in batch:
                    article["title_ja"] = article["title"]
                    article["summary_ja"] = article["summary_en"]

        except Exception as e:
            print(f"  翻訳エラー (バッチ {batch_num}): {e}", file=sys.stderr)
            for article in batch:
                article["title_ja"] = article["title"]
                article["summary_ja"] = article["summary_en"]

        # RPM対策: バッチ間で待機（RPM=5 → 13秒）
        if batch_start + BATCH_SIZE < len(articles):
            time.sleep(BATCH_INTERVAL)

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


def safe_url(url: str) -> str:
    """javascript: 等の危険なスキームを弾く"""
    if url.startswith(("http://", "https://")):
        return url
    return "#"


def generate_html(articles: list[dict], updated: str) -> None:
    with open(TEMPLATE_FILE, encoding="utf-8") as f:
        template = f.read()

    html = template.replace("{{UPDATED}}", updated)

    # カテゴリラベル
    category_labels = {
        "general": "総合",
        "incident": "インシデント",
        "technical": "技術",
        "research": "リサーチ",
        "cve": "CVE",
    }
    category_colors = {
        "general": "#e74c3c",
        "incident": "#e67e22",
        "technical": "#2980b9",
        "research": "#27ae60",
        "cve": "#8e44ad",
    }

    cards_html = ""
    for article in articles:
        cat = article.get("category", "general")
        label = html_mod.escape(category_labels.get(cat, cat))
        color = category_colors.get(cat, "#666")
        title_ja = html_mod.escape(article["title_ja"] or article["title"])
        summary_ja = html_mod.escape(article["summary_ja"] or article["summary_en"])
        pub = html_mod.escape(article.get("published", ""))
        source = html_mod.escape(article.get("source", ""))
        url = safe_url(article.get("url", "#"))

        cards_html += f"""
        <article class="news-card" data-category="{html_mod.escape(cat)}" style="border-top-color:{color}">
          <div class="card-meta">
            <span class="category-badge" style="background:{color}">{label}</span>
            <span class="source">{source}</span>
            <span class="date">{pub}</span>
          </div>
          <h2 class="card-title">
            <a href="{url}" target="_blank" rel="noopener noreferrer">{title_ja}</a>
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
