import os
import json
import random
import re
import html
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from pydantic import BaseModel


# ============================================================
# 基本設定
# ============================================================

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SERPAPI_KEY = os.environ["SERPAPI_KEY"]
PUSHOVER_USER_KEY = os.environ["PUSHOVER_USER_KEY"]
PUSHOVER_API_TOKEN = os.environ["PUSHOVER_API_TOKEN"]

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

LOG_FILE = Path("ai_small_business_log.json")
DOCS_DIR = Path("docs")
ARTICLES_DIR = DOCS_DIR / "articles"
INDEX_FILE = DOCS_DIR / "index.html"

SEARCHES_PER_DAY = 3
RESULTS_PER_SEARCH = 10
MAX_CANDIDATES = 20
PAST_LOG_LIMIT = 100
MAX_SOURCE_CHARS = 18000

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AI-Small-Business-Daily/1.0; "
        "+https://github.com/)"
    )
}


# ============================================================
# 検索ワード
# ============================================================

SEARCH_QUERIES = [
    '"small business" AI revenue case study',
    '"small business" AI increased sales',
    '"small business" "generative AI" revenue',
    '"small business owner" AI automation revenue',
    '"AI-powered" small business revenue',
    'solopreneur AI business revenue',
    '"small company" AI revenue case study',
    '"small business" ChatGPT revenue',
    '"small business" AI profit case study',
    '"entrepreneur" AI business revenue',
    '"small business" AI sales growth',
    '"small business" AI success story revenue',
]


# ============================================================
# OpenAI Structured Outputs
# ============================================================

class SelectedCase(BaseModel):
    selected_id: int
    reason: str


class ArticleData(BaseModel):
    headline: str
    company_name: str
    business_type: str
    company_size: str
    overview: str
    challenge: str
    ai_usage: str
    financial_result: str
    why_it_worked: str
    lessons: str
    application_ideas: str
    limitations: str


# ============================================================
# ユーティリティ
# ============================================================

def now_jst():
    # GitHub ActionsはUTCで動くため、JST (+9) に合わせる
    from datetime import timezone, timedelta
    return datetime.now(timezone(timedelta(hours=9)))


def normalize_url(url):
    try:
        parts = urlsplit(url)
        return urlunsplit(
            (
                parts.scheme.lower(),
                parts.netloc.lower(),
                parts.path.rstrip("/"),
                "",
                "",
            )
        )
    except Exception:
        return url


def clean_one_line(text):
    return re.sub(r"\s+", " ", text or "").strip()


def safe_html(text):
    return html.escape(text or "").replace("\n", "<br>")


def get_pages_base_url():
    """
    GitHub Actions上では GITHUB_REPOSITORY = owner/repository が自動設定される。
    必要なら PAGES_BASE_URL 環境変数で上書き可能。
    """
    custom = os.getenv("PAGES_BASE_URL", "").strip().rstrip("/")
    if custom:
        return custom

    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    if "/" not in repo:
        raise RuntimeError(
            "GitHub Pages URLを決定できません。"
            "GitHub Actions上で実行するか、PAGES_BASE_URLを設定してください。"
        )

    owner, repo_name = repo.split("/", 1)
    return f"https://{owner}.github.io/{repo_name}"


# ============================================================
# ログ
# ============================================================

def load_log():
    if not LOG_FILE.exists():
        return []

    try:
        data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"ログ読み込みエラー: {e}")
        return []


def save_log(log):
    LOG_FILE.write_text(
        json.dumps(log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# SerpAPI
# ============================================================

def search_serpapi(query):
    print(f"SerpAPI検索: {query}")

    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "hl": "en",
        "gl": "us",
        "safe": "active",
        "num": RESULTS_PER_SEARCH,
    }

    response = requests.get(
        "https://serpapi.com/search",
        params=params,
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()

    if data.get("error"):
        raise RuntimeError(f"SerpAPIエラー: {data['error']}")

    results = []

    for item in data.get("organic_results", []):
        title = clean_one_line(item.get("title"))
        link = clean_one_line(item.get("link"))
        snippet = clean_one_line(item.get("snippet"))

        if not title or not link.startswith("http"):
            continue

        results.append(
            {
                "title": title,
                "url": link,
                "snippet": snippet,
                "query": query,
            }
        )

    return results


REVENUE_KEYWORDS = [
    "revenue", "sales", "profit", "profitable", "income",
    "earn", "earned", "earning", "growth", "customers",
    "orders", "conversion", "roi", "saved", "savings",
    "cost reduction", "$", "%"
]


def revenue_score(candidate):
    text = (
        candidate.get("title", "") + " " + candidate.get("snippet", "")
    ).lower()
    return sum(1 for word in REVENUE_KEYWORDS if word in text)


def get_candidates(log):
    today = now_jst().strftime("%Y-%m-%d")
    rng = random.Random(today)

    queries = rng.sample(
        SEARCH_QUERIES,
        min(SEARCHES_PER_DAY, len(SEARCH_QUERIES)),
    )

    past_urls = {
        normalize_url(item.get("url", ""))
        for item in log
        if item.get("url")
    }

    all_results = []

    for query in queries:
        try:
            all_results.extend(search_serpapi(query))
        except Exception as e:
            print(f"検索失敗: {query}: {e}")

    unique_results = []
    seen = set()

    for item in all_results:
        normalized = normalize_url(item["url"])

        if normalized in past_urls or normalized in seen:
            continue

        seen.add(normalized)
        item["revenue_score"] = revenue_score(item)
        unique_results.append(item)

    unique_results.sort(
        key=lambda x: x["revenue_score"],
        reverse=True,
    )

    return unique_results[:MAX_CANDIDATES]


# ============================================================
# OpenAI：候補選定
# ============================================================

def select_case(candidates, log):
    if not candidates:
        raise RuntimeError("未使用の記事候補が見つかりませんでした。")

    client = OpenAI(api_key=OPENAI_API_KEY)

    candidates_text = "\n\n".join(
        f"""候補ID: {i}
タイトル: {item['title']}
URL: {item['url']}
検索結果の説明: {item['snippet']}
収益関連スコア: {item['revenue_score']}"""
        for i, item in enumerate(candidates)
    )

    past_text = "\n".join(
        f"- {item.get('company_name', '')} / {item.get('title', '')}"
        for item in log[-PAST_LOG_LIMIT:]
    ) or "まだありません。"

    prompt = f"""
以下のSerpAPI検索結果から、
「AIを活用したスモールビジネスの事例」として
最も価値の高い候補を1件選んでください。

条件:
- 業種は問わない
- 実在する事業者
- 個人事業も可
- 従業員1～100名程度の規模を優先
- 売上、利益、受注、顧客獲得、コスト削減など、
  収益につながる成果が確認できる事例を優先
- 単なるAI導入ニュースではなく、具体的な活用が分かるものを優先
- 検索結果だけで会社規模が断定できなくても候補にはできるが、
  小規模事業者らしい根拠があるものを優先
- 過去に紹介した企業・事例と実質的に重複するものは避ける
- URLを新しく作らない

過去に紹介した事例:
{past_text}

今回の候補:
{candidates_text}
"""

    response = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "あなたはスモールビジネスのAI活用事例を選定するリサーチャーです。"
                    "与えられた検索結果だけを根拠に選定してください。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        text_format=SelectedCase,
        store=False,
    )

    result = response.output_parsed

    if result is None:
        raise RuntimeError("OpenAIから選定結果を取得できませんでした。")

    if not 0 <= result.selected_id < len(candidates):
        raise RuntimeError(f"不正な候補ID: {result.selected_id}")

    return candidates[result.selected_id], result.reason


# ============================================================
# 参考記事本文の取得
# ============================================================

def fetch_article_text(url):
    """
    選ばれた記事を取得し、HTMLから本文候補テキストを抽出する。
    robots/paywall/JSサイト等で取得できない場合は空文字を返す。
    """
    try:
        response = requests.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=30,
            allow_redirects=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(
            ["script", "style", "nav", "footer", "header",
             "form", "noscript", "svg", "aside"]
        ):
            tag.decompose()

        # article/mainがあれば優先
        root = soup.find("article") or soup.find("main") or soup.body
        if root is None:
            return ""

        paragraphs = []

        for element in root.find_all(["h1", "h2", "h3", "p", "li"]):
            text = clean_one_line(element.get_text(" ", strip=True))

            # 短すぎるUI文字列等は除外
            if len(text) < 35 and element.name not in {"h1", "h2", "h3"}:
                continue

            paragraphs.append(text)

        extracted = "\n".join(paragraphs)

        if len(extracted) < 300:
            return ""

        return extracted[:MAX_SOURCE_CHARS]

    except Exception as e:
        print(f"参考記事本文を取得できませんでした: {e}")
        return ""


# ============================================================
# OpenAI：長文記事生成
# ============================================================

def generate_article(candidate, source_text):
    client = OpenAI(api_key=OPENAI_API_KEY)

    source_section = (
        source_text
        if source_text
        else "参考記事本文は取得できませんでした。検索結果の説明のみ利用できます。"
    )

    prompt = f"""
以下の情報だけを根拠として、日本語で
「AIを活用したスモールビジネスの事例」の解説記事を作成してください。

目安は1,500～2,500字程度です。
ただし、根拠情報が少ない場合は無理に長くせず、
事実を水増ししないことを最優先してください。

【検索結果】
タイトル: {candidate['title']}
URL: {candidate['url']}
検索結果の説明: {candidate['snippet']}

【取得できた参考記事本文】
{source_section}

重要ルール:
- 上記情報にない事実を作らない
- 従業員数、売上、利益、顧客数、割合などの数字を推測しない
- 「AI導入によって収益が増えた」と因果関係が確認できない場合は断定しない
- 会社規模が確認できない場合はその旨を書く
- 収益額が確認できない場合はその旨を書く
- 外部知識で穴埋めしない
- URLは本文内に生成しない
- 日本の小規模事業者への応用部分は分析・提案なので、
  事実部分と区別して書く
- 読み物として自然で、専門用語を使いすぎない
- 同じ説明を繰り返して文字数を増やさない

各項目:
headline:
記事タイトル。会社名とAI活用の特徴が分かるもの。

company_name:
事業者名。不明なら「記事から確認できません」。

business_type:
どのような事業か。

company_size:
分かる範囲の規模感。確認できない場合は明記。

overview:
事例の全体像を2～4段落程度。

challenge:
AI活用前の課題。記事から明確でない場合はその旨を書く。

ai_usage:
AIを何に、どう使ったのかを具体的に。

financial_result:
売上、利益、受注、顧客獲得、コスト削減など、
収益とのつながり。確認できた内容だけを書く。

why_it_worked:
なぜこの活用が有効だったと考えられるか。
ここは「記事内容からの分析」と分かる書き方にする。

lessons:
他のスモールビジネスが学べるポイント。

application_ideas:
日本の小規模企業、士業、会計事務所、保険代理店などに
応用するとしたらどんな方法があるか。
これは提案・考察として書く。

limitations:
記事から確認できない点や、評価時の注意点。
"""

    response = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "あなたは実在するスモールビジネスのAI活用事例を"
                    "根拠に基づいて解説する日本語ライターです。"
                    "確認できないことを推測して事実として書いてはいけません。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        text_format=ArticleData,
        store=False,
    )

    article = response.output_parsed

    if article is None:
        raise RuntimeError("OpenAIから記事データを取得できませんでした。")

    return article


# ============================================================
# HTML生成
# ============================================================

BASE_CSS = """
:root {
  --bg: #f5f5f2;
  --card: #ffffff;
  --text: #202124;
  --sub: #666a70;
  --line: #dedfd9;
  --accent: #1f5d50;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family:
    -apple-system, BlinkMacSystemFont, "Segoe UI",
    "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
  line-height: 1.9;
}
a { color: var(--accent); }
.wrap {
  width: min(880px, calc(100% - 32px));
  margin: 0 auto;
}
.site-header {
  padding: 40px 0 24px;
  border-bottom: 1px solid var(--line);
}
.brand {
  margin: 0;
  font-size: 14px;
  letter-spacing: .16em;
  font-weight: 700;
}
.tagline {
  margin: 7px 0 0;
  color: var(--sub);
  font-size: 14px;
}
main { padding: 34px 0 64px; }
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: clamp(22px, 5vw, 46px);
  margin-bottom: 24px;
}
.date {
  color: var(--sub);
  font-size: 13px;
  letter-spacing: .05em;
}
h1 {
  font-size: clamp(28px, 5vw, 42px);
  line-height: 1.35;
  margin: 12px 0 20px;
}
h2 {
  font-size: 20px;
  margin: 38px 0 12px;
  padding-top: 6px;
  border-top: 1px solid var(--line);
}
p { margin: 0 0 16px; }
.meta {
  padding: 16px 18px;
  background: #f8f8f5;
  border-radius: 10px;
  margin: 22px 0;
  font-size: 14px;
}
.note {
  color: var(--sub);
  font-size: 14px;
}
.source {
  word-break: break-all;
}
.article-list {
  list-style: none;
  padding: 0;
  margin: 0;
}
.article-list li {
  padding: 20px 0;
  border-top: 1px solid var(--line);
}
.article-list li:first-child { border-top: 0; }
.article-list a {
  font-size: 18px;
  font-weight: 700;
  text-decoration: none;
}
.article-list a:hover { text-decoration: underline; }
footer {
  padding: 30px 0 50px;
  color: var(--sub);
  font-size: 12px;
}
"""


def paragraphize(text):
    """
    Structured Output内の改行を段落としてHTML化。
    """
    chunks = [
        clean_one_line(x)
        for x in re.split(r"\n+", text or "")
        if clean_one_line(x)
    ]

    if not chunks:
        return "<p>記事から確認できません。</p>"

    return "".join(f"<p>{html.escape(x)}</p>" for x in chunks)


def write_article_html(article, candidate, date_str, article_filename):
    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

    article_path = ARTICLES_DIR / article_filename

    page = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(article.headline)} | AI Small Business Daily</title>
<meta name="description" content="{html.escape(clean_one_line(article.overview)[:150])}">
<style>{BASE_CSS}</style>
</head>
<body>
<header class="site-header">
  <div class="wrap">
    <p class="brand"><a href="../index.html" style="text-decoration:none;color:inherit;">AI SMALL BUSINESS DAILY</a></p>
    <p class="tagline">AIを活用して成果を生み出す、小さなビジネスの事例集。</p>
  </div>
</header>

<main class="wrap">
  <article class="card">
    <div class="date">{html.escape(date_str)}</div>
    <h1>{html.escape(article.headline)}</h1>

    <div class="meta">
      <strong>事業者：</strong>{html.escape(article.company_name)}<br>
      <strong>業種：</strong>{html.escape(article.business_type)}<br>
      <strong>規模：</strong>{html.escape(article.company_size)}
    </div>

    <h2>今回の事例</h2>
    {paragraphize(article.overview)}

    <h2>どんな課題があったのか</h2>
    {paragraphize(article.challenge)}

    <h2>AIをどう活用したのか</h2>
    {paragraphize(article.ai_usage)}

    <h2>どう収益につながったのか</h2>
    {paragraphize(article.financial_result)}

    <h2>なぜこのAI活用が有効だったのか</h2>
    {paragraphize(article.why_it_worked)}

    <h2>この事例から学べること</h2>
    {paragraphize(article.lessons)}

    <h2>日本のスモールビジネスならどう応用できる？</h2>
    {paragraphize(article.application_ideas)}

    <h2>確認できない点・注意点</h2>
    {paragraphize(article.limitations)}

    <h2>参考記事</h2>
    <p>{html.escape(candidate['title'])}</p>
    <p class="source">
      <a href="{html.escape(candidate['url'], quote=True)}"
         target="_blank" rel="noopener noreferrer">
        {html.escape(candidate['url'])}
      </a>
    </p>

    <p class="note">
      ※この記事は公開されている参考記事をAIで整理・分析したものです。
      確認できない情報は推測せず、応用アイデアは考察として記載しています。
    </p>
  </article>
</main>

<footer>
  <div class="wrap">AI Small Business Daily</div>
</footer>
</body>
</html>
"""

    article_path.write_text(page, encoding="utf-8")


def write_index_html(log):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    entries = list(reversed(log))

    if entries:
        list_html = "\n".join(
            f"""<li>
  <div class="date">{html.escape(item.get('date_display', item.get('date', '')))}</div>
  <a href="articles/{html.escape(item['article_filename'], quote=True)}">
    {html.escape(item.get('headline') or item.get('title', '記事を読む'))}
  </a>
  <div class="note">{html.escape(item.get('company_name', ''))}</div>
</li>"""
            for item in entries
            if item.get("article_filename")
        )
    else:
        list_html = "<li>まだ記事はありません。</li>"

    page = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Small Business Daily</title>
<meta name="description" content="AIを活用して成果を生み出す世界のスモールビジネス事例を紹介します。">
<style>{BASE_CSS}</style>
</head>
<body>
<header class="site-header">
  <div class="wrap">
    <p class="brand">AI SMALL BUSINESS DAILY</p>
    <p class="tagline">AIを活用して成果を生み出す、小さなビジネスの事例集。</p>
  </div>
</header>

<main class="wrap">
  <section class="card">
    <h1>小さなビジネスの<br>AI活用を、毎日のヒントに。</h1>
    <p>
      世界のスモールビジネスから、AIを実際の事業に取り入れ、
      売上・顧客獲得・コスト削減などにつなげている事例を紹介します。
    </p>
  </section>

  <section class="card">
    <h2 style="margin-top:0;border-top:0;">記事一覧</h2>
    <ul class="article-list">
      {list_html}
    </ul>
  </section>
</main>

<footer>
  <div class="wrap">AI Small Business Daily</div>
</footer>
</body>
</html>
"""

    INDEX_FILE.write_text(page, encoding="utf-8")


# ============================================================
# Pushover
# ============================================================

def send_pushover(article, article_url):
    message = (
        "今日のAI活用事例を更新しました。\n\n"
        f"{article.headline}\n\n"
        "下の「今日の記事を読む」から開けます。"
    )

    response = requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": PUSHOVER_API_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "title": "AI Small Business Daily",
            "message": message,
            "url": article_url,
            "url_title": "今日の記事を読む",
        },
        timeout=30,
    )

    response.raise_for_status()
    data = response.json()

    if data.get("status") != 1:
        raise RuntimeError(f"Pushover送信エラー: {data}")

    print("Pushover送信成功")


# ============================================================
# メイン
# ============================================================

def main():
    print("=== AI Small Business Daily ===")

    log = load_log()
    print(f"過去ログ: {len(log)}件")

    candidates = get_candidates(log)
    print(f"未使用候補: {len(candidates)}件")

    if not candidates:
        raise RuntimeError("未使用候補がありません。")

    for i, item in enumerate(candidates):
        print(f"[{i}] score={item['revenue_score']} {item['title']}")

    candidate, selection_reason = select_case(candidates, log)

    print(f"\n選択記事: {candidate['title']}")
    print(candidate["url"])
    print(f"選定理由: {selection_reason}")

    source_text = fetch_article_text(candidate["url"])
    print(f"取得本文文字数: {len(source_text)}")

    article = generate_article(candidate, source_text)

    jst = now_jst()
    date_iso = jst.strftime("%Y-%m-%d")
    date_display = jst.strftime("%Y.%m.%d")
    article_filename = f"{date_iso}.html"

    pages_base_url = get_pages_base_url()
    article_url = f"{pages_base_url}/articles/{article_filename}"

    # 先にHTMLとログを生成
    write_article_html(
        article,
        candidate,
        date_display,
        article_filename,
    )

    log_entry = {
        "date": date_iso,
        "date_display": date_display,
        "company_name": article.company_name,
        "headline": article.headline,
        "title": candidate["title"],
        "url": candidate["url"],
        "article_filename": article_filename,
        "article_url": article_url,
        "search_query": candidate["query"],
        "selection_reason": selection_reason,
    }

    # 同日の手動再実行時は同じ日付の記事を置き換える
    log = [
        item for item in log
        if item.get("date") != date_iso
    ]
    log.append(log_entry)

    save_log(log)
    write_index_html(log)

    print(f"記事生成完了: {article_url}")

    # IMPORTANT:
    # GitHub Pagesへの実際の反映は、このPython終了後に
    # workflowがcommit/pushして行う。
    # Pushover通知は workflow 側から --notify-only で再実行する。
    return article_url


def notify_only():
    log = load_log()

    if not log:
        raise RuntimeError("通知対象の記事ログがありません。")

    latest = log[-1]

    class MinimalArticle:
        headline = latest.get("headline", "今日のAI活用事例")

    send_pushover(
        MinimalArticle(),
        latest["article_url"],
    )


if __name__ == "__main__":
    try:
        if "--notify-only" in sys.argv:
            notify_only()
        else:
            main()
    except Exception as e:
        print(f"\nエラー: {e}")
        raise
