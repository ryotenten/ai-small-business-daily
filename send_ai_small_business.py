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
MIN_ADOPTION_SCORE = 70
TOP_CANDIDATES_TO_EVALUATE = 5

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
    '"AI workflow" automation "how I"',
    '"I automated" ChatGPT workflow',
    '"I built" AI automation workflow',
    '"how I use ChatGPT" automate work',
    '"OpenAI API" workflow automation case study',
    '"AI automation" Gmail spreadsheet workflow',
    '"AI workflow" Excel automation',
    '"AI workflow" small business operations',
    'solopreneur AI workflow automation',
    '"AI agent" practical workflow business',
    'site:reddit.com AI workflow automation work',
    'site:github.com AI automation workflow OpenAI',
    'site:news.ycombinator.com AI automation workflow',
    'site:qiita.com AI OpenAI 自動化 業務',
    'site:zenn.dev AI OpenAI 自動化 業務',
]



# ============================================================
# OpenAI Structured Outputs
# ============================================================

class ShortlistSelection(BaseModel):
    selected_ids: list[int]
    reason: str


class CaseEvaluation(BaseModel):
    adopt: bool
    total_score: int
    specificity_score: int
    reproducibility_score: int
    usefulness_score: int
    applicability_score: int
    reliability_score: int
    workflow_signature: str
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


WORKFLOW_KEYWORDS = [
    "workflow", "automate", "automation", "integrate", "integration",
    "api", "python", "zapier", "make.com", "n8n", "gmail", "slack",
    "notion", "spreadsheet", "excel", "google sheets", "extract",
    "classify", "agent", "pipeline", "process", "saved time",
    "i automated", "i built", "how i use", "case study", "hours saved",
    "before and after", "before/after"
]

# 比較記事・SEO記事・一般論は、実践ワークフローである可能性が低いため減点する。
SEO_LIKE_KEYWORDS = [
    "best ai", "best tools", "top 5", "top 10", "top 20",
    "platforms", "what is", "ultimate guide", "complete guide",
    "tools for", "tested & compared", "tested and compared",
    "comparison", "buyers guide", "buyer's guide"
]


def workflow_score(candidate):
    text = (
        candidate.get("title", "") + " " + candidate.get("snippet", "")
    ).lower()

    positive = sum(1 for word in WORKFLOW_KEYWORDS if word in text)
    penalty = sum(2 for word in SEO_LIKE_KEYWORDS if word in text)
    return positive - penalty


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
        item["workflow_score"] = workflow_score(item)
        unique_results.append(item)

    unique_results.sort(
        key=lambda x: x["workflow_score"],
        reverse=True,
    )

    return unique_results[:MAX_CANDIDATES]


# ============================================================
# OpenAI：候補選定
# ============================================================

def select_cases(candidates, log):
    """検索結果だけを使い、本文確認する有望候補を最大5件に絞る。"""
    if not candidates:
        raise RuntimeError("未使用の記事候補が見つかりませんでした。")

    client = OpenAI(api_key=OPENAI_API_KEY)

    candidates_text = "\n\n".join(
        f"""候補ID: {i}
タイトル: {item['title']}
URL: {item['url']}
検索結果の説明: {item['snippet']}
ワークフロー関連スコア: {item['workflow_score']}"""
        for i, item in enumerate(candidates)
    )

    past_text = "\n".join(
        f"- {item.get('headline', item.get('title', ''))} / "
        f"ワークフロー: {item.get('workflow_signature', '記録なし')}"
        for item in log[-PAST_LOG_LIMIT:]
    ) or "まだありません。"

    prompt = f"""
以下のSerpAPI検索結果から、本文を詳しく確認する価値が高い候補を
最大{TOP_CANDIDATES_TO_EVALUATE}件、優先順位順に選んでください。

目的は「AIを導入した会社」を集めることではありません。
読者自身が仕事や日常生活で真似・再現・応用できる、具体的なAIワークフローを集めることです。

優先する候補:
- 入力 → AI処理 → 出力 → 人間の最終作業、の流れが想像できる
- AIを何に使ったかが具体的
- 個人、中小企業、少人数チームでも再現できそう
- Gmail、Excel、Google Sheets、PDF、Slack、Notion、Python、API、Zapier、Make、n8n、GitHub Actions等との組み合わせが分かる
- 時間削減、ミス削減、品質向上、顧客対応改善など実務上の価値がある
- 一つの仕組みから他の業務・生活にも応用できそう
- 大企業の事例でも、ワークフロー自体を小規模に再現できるなら候補にしてよい
- 仕事だけでなく個人生活で役立つ実践例も可

強く優先しない候補:
- 「Best tools」「Top 10」「What is ...」「comparison」などの比較・SEO記事
- AI業界ニュース、モデル発表、株価、資金調達、市場規模
- 「AIを導入した」だけで具体的な使い方が分からないもの
- 単なる製品PR
- 一般的な文章生成・要約だけのもの
- 過去事例と本質的に同じワークフロー

この段階では検索結果だけなので、最終採用判定はしません。
「本文を読めば有用な具体例が見つかりそうか」で候補を選んでください。
selected_ids には候補IDを重複なく、優先順位順に最大{TOP_CANDIDATES_TO_EVALUATE}件入れてください。
URLは新しく作らないでください。

過去に保存した事例:
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
                    "あなたは実践的なAIワークフローを発見するリサーチャーです。"
                    "検索結果だけで断定せず、本文確認する価値の高い候補を複数選んでください。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        text_format=ShortlistSelection,
        store=False,
    )

    result = response.output_parsed
    if result is None:
        raise RuntimeError("OpenAIから候補選定結果を取得できませんでした。")

    valid_ids = []
    seen = set()
    for candidate_id in result.selected_ids:
        if not 0 <= candidate_id < len(candidates):
            continue
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        valid_ids.append(candidate_id)
        if len(valid_ids) >= TOP_CANDIDATES_TO_EVALUATE:
            break

    if not valid_ids:
        raise RuntimeError("本文確認対象となる有効な候補IDがありませんでした。")

    return [candidates[i] for i in valid_ids], result.reason


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
# OpenAI：本文を読んだ最終採用判定
# ============================================================

def evaluate_case(candidate, source_text, log):
    client = OpenAI(api_key=OPENAI_API_KEY)

    source_section = (
        source_text
        if source_text
        else "参考記事本文を取得できませんでした。検索結果の説明のみです。"
    )

    past_text = "\n".join(
        f"- {item.get('headline', item.get('title', ''))} / "
        f"ワークフロー: {item.get('workflow_signature', '記録なし')}"
        for item in log[-PAST_LOG_LIMIT:]
    ) or "まだありません。"

    prompt = f"""
以下の候補を「長期的に残す価値のあるAI活用ワークフロー」として採用するか評価してください。

【候補】
タイトル: {candidate['title']}
URL: {candidate['url']}
検索結果の説明: {candidate['snippet']}

【取得できた本文】
{source_section}

【過去に保存した事例】
{past_text}

100点満点で厳しく評価してください。

1. 具体性 0～25点
何を入力し、AIが何をし、何を出力するかが具体的か。

2. 再現性 0～25点
個人や中小企業が、一般的なツールやAPI等で似た仕組みを再現できるか。

3. 実用性 0～25点
時間削減、ミス削減、品質向上、顧客対応改善、意思決定改善などにつながるか。

4. 応用可能性 0～15点
このワークフローの考え方を、別の仕事・業種・個人生活にも横展開しやすいか。
珍しさそのものは重視しない。既視感があっても、具体的で真似しやすく応用範囲が広ければ高得点にしてよい。
ただし、過去事例と本質的に同じワークフローなら重複として減点する。

5. 情報信頼性 0～10点
実際の使い方が本文から確認でき、単なる広告や憶測に偏っていないか。

採用ルール:
- total_score が {MIN_ADOPTION_SCORE} 点以上なら原則 adopt=true
- ただし、AIの具体的な使い方が本文から確認できない場合は70点以上でも adopt=false
- 本文取得に失敗し、snippetだけでは具体的なワークフローを確認できない場合は adopt=false
- AIニュース、モデル発表、資金調達、業界論だけなら adopt=false
- 過去事例と本質的に重複する場合は adopt=false
- 良い候補でなければ、記事を作るために無理に採用しない

workflow_signature は、企業名を使わず、ワークフローの本質を短く表してください。
例: 「問い合わせメール→AIで分類・項目抽出→CRM登録→担当者通知」
"""

    response = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "あなたはAI活用図鑑の編集者です。"
                    "毎日1件埋めることより、具体性・再現性・実用性の高い事例だけ残すことを優先してください。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        text_format=CaseEvaluation,
        store=False,
    )

    result = response.output_parsed
    if result is None:
        raise RuntimeError("OpenAIから採用判定を取得できませんでした。")

    # モデルのtotal_scoreと内訳が食い違った場合は内訳を正とする
    calculated = (
        result.specificity_score
        + result.reproducibility_score
        + result.usefulness_score
        + result.applicability_score
        + result.reliability_score
    )
    result.total_score = calculated

    if calculated < MIN_ADOPTION_SCORE:
        result.adopt = False

    return result


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
以下の情報を根拠として、日本語で「真似できるAI活用ワークフロー」の記事を作成してください。
目的はニュース紹介ではなく、読者が「これなら自分でも使えそう」と思えるAI活用図鑑を作ることです。

【検索結果】
タイトル: {candidate['title']}
URL: {candidate['url']}
検索結果の説明: {candidate['snippet']}

【取得できた参考記事本文】
{source_section}

最重要ルール:
- 元記事で確認できる事実と、あなたが考える応用案を明確に分ける
- 元記事にない運用方法を「実際に行っている」と書かない
- 数字、使用ツール、成果、企業規模などを推測しない
- URLは本文内に新しく生成しない
- 「AIを活用」「効率化した」だけで終わらず、可能な限り入力→AI処理→出力→人間の作業を説明する
- 具体的なフローが不明な部分は「記事からは確認できない」と明記する
- 応用案では、どのデータを使い、AIに何をさせ、どこへ出力するかまで具体化する
- 毎回むりに会計・保険へ結びつけない。適合する場合だけ具体案を書く
- 仕事だけでなく、自然に応用できるなら個人利用も提案する
- 一般論を繰り返して文字数を水増ししない

各項目:
headline:
会社名より「何をAI化したのか」が一目で分かるタイトル。必要なら会社名も含める。

company_name:
実際の事例の企業・人物名。不明なら「記事から確認できません」。

business_type:
業種・用途。

company_size:
確認できる場合のみ。確認できなければその旨を書く。

overview:
最初の2～4文で「何が面倒だったか」「AIで何を変えたか」を説明し、その後に実際の事例概要を書く。

challenge:
AI導入前の課題とBefore。記事から確認できる範囲だけを書く。

ai_usage:
最重要項目。
可能なら次の形で具体的に書く:
入力
↓
AIによる処理
↓
次のシステム・処理
↓
出力
↓
人間による確認・最終作業
さらにAIが分類、抽出、要約、判断補助、文章生成、検索など何を担当しているか説明する。

financial_result:
見出し上は「得られた効果」として使う。
売上だけでなく、時間削減、ミス削減、返信速度、品質、顧客対応など確認できる効果を書く。
数値がなければ作らない。

why_it_worked:
なぜこのワークフローが有効だったと考えられるか。元記事の事実ではなく分析なら、その旨が分かるようにする。

lessons:
この事例の再利用可能なポイント。企業固有事情ではなく、他の仕事にも移植できる考え方を中心にする。

application_ideas:
「自分で作るなら」を中心に、実装案を具体化する。
例:
1. データを取得
2. AIへ渡す
3. AIで処理
4. Excel/DB/Notion等へ保存
5. Slack/Pushover/メール等で通知
6. 人間が確認

使えそうなものがあれば ChatGPT、OpenAI API、Python、Excel、Google Sheets、Gmail、GitHub Actions、Zapier、Make、Slack、Pushover 等を挙げる。
その後、仕事への応用例を2～4個、自然に可能ならプライベート応用を1～3個書く。
「ChatGPTで要約する」のような抽象論ではなく、入力→処理→出力まで示す。

limitations:
元記事から確認できないこと、導入上の注意、個人情報・機密情報・誤判定など人間確認が必要な点を書く。
"""

    response = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "あなたは実践的なAI活用図鑑を作る日本語編集者です。"
                    "ニュース性より具体性・再現性・実用性を重視し、事実と提案を混同しません。"
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
    <p class="tagline">仕事や暮らしで真似できる、実践的なAIワークフロー集。</p>
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

    <h2>どんな効果があったのか</h2>
    {paragraphize(article.financial_result)}

    <h2>なぜこのAI活用が有効だったのか</h2>
    {paragraphize(article.why_it_worked)}

    <h2>この事例から学べること</h2>
    {paragraphize(article.lessons)}

    <h2>自分で作るなら？・どう応用できる？</h2>
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
<meta name="description" content="仕事や暮らしで再現・応用できるAIワークフローを紹介します。">
<style>{BASE_CSS}</style>
</head>
<body>
<header class="site-header">
  <div class="wrap">
    <p class="brand">AI SMALL BUSINESS DAILY</p>
    <p class="tagline">仕事や暮らしで真似できる、実践的なAIワークフロー集。</p>
  </div>
</header>

<main class="wrap">
  <section class="card">
    <h1>真似できるAI活用を、<br>毎日のヒントに。</h1>
    <p>
      世界の実践例から、入力→AI処理→出力が見える具体的なワークフローを選び、
      個人や中小企業でも再現・応用できる形で紹介します。
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
    print("=== AI Practical Workflow Daily ===")

    log = load_log()
    print(f"過去ログ: {len(log)}件")

    candidates = get_candidates(log)
    print(f"未使用候補: {len(candidates)}件")

    if not candidates:
        print("未使用候補がありません。本日は記事を作成しません。")
        return None

    for i, item in enumerate(candidates):
        print(f"[{i}] workflow_score={item['workflow_score']} {item['title']}")

    shortlisted, shortlist_reason = select_cases(candidates, log)

    print(f"\n本文確認候補: {len(shortlisted)}件")
    print(f"候補選定理由: {shortlist_reason}")

    evaluated = []

    for rank, candidate in enumerate(shortlisted, start=1):
        print("\n" + "=" * 60)
        print(f"候補{rank}: {candidate['title']}")
        print(candidate["url"])

        source_text = fetch_article_text(candidate["url"])
        print(f"取得本文文字数: {len(source_text)}")

        # 本文が十分取れなくてもevaluate_case側でsnippetを踏まえて不採用判定できる。
        # 1件の取得失敗で全体を止めない。
        try:
            evaluation = evaluate_case(candidate, source_text, log)
        except Exception as e:
            print(f"採用評価に失敗したためこの候補をスキップします: {e}")
            continue

        print(f"採用判定: {evaluation.adopt}")
        print(f"総合スコア: {evaluation.total_score}/100")
        print(
            "内訳: "
            f"具体性={evaluation.specificity_score}/25, "
            f"再現性={evaluation.reproducibility_score}/25, "
            f"実用性={evaluation.usefulness_score}/25, "
            f"応用可能性={evaluation.applicability_score}/15, "
            f"信頼性={evaluation.reliability_score}/10"
        )
        print(f"ワークフロー: {evaluation.workflow_signature}")
        print(f"判定理由: {evaluation.reason}")

        evaluated.append({
            "candidate": candidate,
            "source_text": source_text,
            "evaluation": evaluation,
        })

    if not evaluated:
        print("評価できる候補がありませんでした。本日は記事を生成しません。")
        return None

    # 採用可の候補だけに絞り、総合点が最も高いものを選ぶ。
    adopted = [item for item in evaluated if item["evaluation"].adopt]

    print("\n" + "=" * 60)
    print("本日の評価結果")
    for item in sorted(evaluated, key=lambda x: x["evaluation"].total_score, reverse=True):
        ev = item["evaluation"]
        print(
            f"- {ev.total_score:3d}点 / {'採用' if ev.adopt else '不採用'} / "
            f"{item['candidate']['title']}"
        )

    if not adopted:
        best_rejected = max(evaluated, key=lambda x: x["evaluation"].total_score)
        print(
            f"最高点は {best_rejected['evaluation'].total_score}/100 でした。"
            "70点以上の候補がないため、本日は記事を生成・保存・通知しません。"
        )
        return None

    best = max(adopted, key=lambda x: x["evaluation"].total_score)
    candidate = best["candidate"]
    source_text = best["source_text"]
    evaluation = best["evaluation"]

    print("\n採用記事を決定しました。")
    print(f"タイトル: {candidate['title']}")
    print(f"総合スコア: {evaluation.total_score}/100")
    print(f"ワークフロー: {evaluation.workflow_signature}")

    article = generate_article(candidate, source_text)

    jst = now_jst()
    date_iso = jst.strftime("%Y-%m-%d")
    date_display = jst.strftime("%Y.%m.%d")
    article_filename = f"{date_iso}.html"

    pages_base_url = get_pages_base_url()
    article_url = f"{pages_base_url}/articles/{article_filename}"

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
        "selection_reason": shortlist_reason,
        "adoption_score": evaluation.total_score,
        "workflow_signature": evaluation.workflow_signature,
        "evaluation_reason": evaluation.reason,
        "score_detail": {
            "specificity": evaluation.specificity_score,
            "reproducibility": evaluation.reproducibility_score,
            "usefulness": evaluation.usefulness_score,
            "applicability": evaluation.applicability_score,
            "reliability": evaluation.reliability_score,
        },
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

    # GitHub Pagesへの実際の反映は、このPython終了後にworkflowがcommit/pushして行う。
    # Pushover通知は workflow 側から --notify-only で再実行する。
    return article_url


def notify_only():
    log = load_log()

    if not log:
        print("通知対象の記事ログがありません。通知しません。")
        return

    latest = log[-1]
    today = now_jst().strftime("%Y-%m-%d")

    # 本日新しい記事が採用されなかった場合、前日の記事を再通知しない
    if latest.get("date") != today:
        print("本日採用された新規記事はありません。Pushover通知をスキップします。")
        return

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
