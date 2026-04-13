import csv
import json
import os
import re
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

INPUT_CSV = os.getenv("INPUT_CSV", "tgstat_channels.csv")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "tgstat_channels_postprocessed.csv")
OUTPUT_JSONL = os.getenv("OUTPUT_JSONL", "tgstat_channels_postprocessed.jsonl")
STATE_JSON = os.getenv("STATE_JSON", "tgstat_postprocess_state.json")
OUTPUT_STORAGE_STATE = os.getenv("OUTPUT_STORAGE_STATE", "storage_state_postprocess.json")
OUTPUT_ERRORS_LOG = os.getenv("POSTPROCESS_ERRORS_LOG", "postprocess_errors.log")
OUTPUT_DEBUG_LOG = os.getenv("POSTPROCESS_DEBUG_LOG", "postprocess_debug.log")

REQUEST_DELAY = 1.0
PLAYWRIGHT_HEADLESS = "false"
PLAYWRIGHT_MANUAL_CHALLENGE = "true"

BASE_FIELDS = [
    "source_page", "source_page_type", "channel_url", "username", "title", "description",
    "subscribers", "avg_reach", "citation_index", "country", "category", "tag_group",
    "rank_subscribers", "last_seen_text", "last_seen_minutes", "title_length",
    "description_length", "word_count", "has_contact", "has_link", "has_ad_label",
    "has_18plus", "language_hint", "engagement_ratio", "text_description",
]

EXTRA_FIELDS = [
    "subscribers_today_delta",
    "subscribers_week_delta",
    "subscribers_month_delta",
    "subscribers_baseline_date",
    "subscribers_baseline_value",

    "subscriptions_24h",
    "unsubscriptions_24h",
    "net_subscriptions_24h",

    "mentions_channels",
    "mentions_total",
    "reposts_total",
    "citation_index_date",
    "citation_index_baseline_value",

    "avg_reach_1_post",
    "err_percent",
    "err24_percent",
    "avg_reach_date",
    "avg_reach_baseline_value",

    "avg_ad_reach_1_post",
    "avg_ad_reach_12h",
    "avg_ad_reach_24h",
    "avg_ad_reach_48h",

    "stories_avg_reach",
    "stories_reach_12h",
    "stories_reach_24h",
    "stories_reach_48h",
    "stories_boosts",
    "stories_reactions",
    "stories_forwards",

    "channel_age_text",
    "channel_created_date",
    "added_to_tgstat_date",

    "posts_total",
    "posts_yesterday",
    "posts_week",
    "posts_month",

    "readers_percent",
    "readers_24h_percent",

    "er_percent",
    "er_forward_count",
    "er_comment_count",
    "er_reaction_count",
    "er_date",
    "er_baseline_value",

    "male_percent",
    "female_percent",

    "avg_reach_to_subscribers_ratio",
    "ad_reach_to_subscribers_ratio",
    "stories_reach_to_subscribers_ratio",
    "mentions_per_1000_subscribers",
    "reposts_per_1000_subscribers",
    "posts_per_week_estimated",
]

ALL_FIELDS = BASE_FIELDS + EXTRA_FIELDS


def load_env():
    load_dotenv()


def load_int_from_env(env_var_name, default_value):
    load_env()
    raw = os.getenv(env_var_name, "").strip()
    if not raw:
        return default_value
    try:
        return int(raw)
    except ValueError:
        return default_value


def load_float_from_env(env_var_name, default_value):
    load_env()
    raw = os.getenv(env_var_name, "").strip()
    if not raw:
        return default_value
    try:
        return float(raw)
    except ValueError:
        return default_value


def load_bool_from_env(env_var_name, default_value=False):
    load_env()
    raw = os.getenv(env_var_name, "").strip().lower()
    if not raw:
        return default_value
    return raw in {"1", "true", "yes", "y", "on"}


def load_proxy_from_env():
    load_env()
    proxy = os.getenv("PLAYWRIGHT_PROXY") or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    if not proxy:
        return None
    return {"server": proxy.strip()}


def load_json_from_env_path(env_var_name, default=None):
    load_env()
    raw_path = os.getenv(env_var_name, "").strip()
    if not raw_path:
        return default
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл из переменной {env_var_name} не найден: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_headers_from_env_path(env_var_name="PLAYWRIGHT_HEADERS_PATH"):
    default_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    parsed = load_json_from_env_path(env_var_name, default={}) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"{env_var_name} должен указывать на JSON-объект")
    normalized = {str(k): str(v) for k, v in parsed.items() if k is not None and v is not None}
    return {**default_headers, **normalized}


def split_user_agent_from_headers(headers: dict):
    headers_copy = dict(headers)
    user_agent = headers_copy.get("User-Agent") or headers_copy.get("user-agent")
    headers_copy.pop("User-Agent", None)
    headers_copy.pop("user-agent", None)
    return user_agent, headers_copy


def load_cookies_from_env_path(env_var_name="PLAYWRIGHT_COOKIES_PATH", default_domain="tgstat.ru"):
    parsed = load_json_from_env_path(env_var_name, default={})
    if parsed is None:
        return []
    normalized = []
    if isinstance(parsed, dict):
        for name, value in parsed.items():
            if name is None or value is None:
                continue
            normalized.append({"name": str(name), "value": str(value), "domain": default_domain, "path": "/"})
        return normalized
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            value = item.get("value")
            if not name or value is None:
                continue
            cookie = {"name": str(name), "value": str(value)}
            if item.get("url"):
                cookie["url"] = str(item["url"])
            else:
                cookie["domain"] = str(item.get("domain", default_domain))
                cookie["path"] = str(item.get("path", "/"))
            if "httpOnly" in item:
                cookie["httpOnly"] = bool(item["httpOnly"])
            if "secure" in item:
                cookie["secure"] = bool(item["secure"])
            if "sameSite" in item and item["sameSite"] in ("Strict", "Lax", "None"):
                cookie["sameSite"] = item["sameSite"]
            if "expires" in item and item["expires"] is not None:
                cookie["expires"] = item["expires"]
            normalized.append(cookie)
        return normalized
    raise ValueError(f"{env_var_name} должен указывать либо на JSON-объект cookies, либо на JSON-массив cookie-объектов")


def debug_log(message, path=OUTPUT_DEBUG_LOG):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    print(line)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_error(message, path=OUTPUT_ERRORS_LOG):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def atomic_write_json(path, data, retries=12, retry_delay=0.25):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    last_error = None
    for _ in range(retries):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as e:
            last_error = e
            time.sleep(retry_delay)
            try:
                if path.exists():
                    os.remove(path)
            except (PermissionError, FileNotFoundError):
                pass
            time.sleep(retry_delay)
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass
    raise PermissionError(f"Не удалось атомарно заменить {tmp_path} -> {path}. Последняя ошибка: {last_error}")


def normalize_url(url):
    if not url:
        return ""
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl()


def is_cloudflare_challenge_html(html):
    if not html:
        return False
    low = html.lower()
    markers = [
        "cloudflare",
        "challenge-platform",
        "checking your browser",
        "выполнение проверки безопасности",
        "проверяет, что вы не бот",
        "challenges.cloudflare.com",
        "cf-chl-",
        "turnstile",
    ]
    return sum(marker in low for marker in markers) >= 2


def humanize_page(page):
    try:
        page.wait_for_timeout(1200)
        page.mouse.move(200, 200)
        page.wait_for_timeout(400)
        page.mouse.wheel(0, 600)
        page.wait_for_timeout(800)
        page.mouse.wheel(0, -200)
        page.wait_for_timeout(400)
    except Exception:
        pass


def safe_int_from_text(value):
    if value is None:
        return None
    raw = str(value).strip().replace("\u00a0", " ").replace(",", ".")
    if not raw:
        return None
    raw_lower = raw.lower()
    multiplier = 1
    if "млн" in raw_lower or raw_lower.endswith("m"):
        multiplier = 1_000_000
    elif "тыс" in raw_lower or raw_lower.endswith("k"):
        multiplier = 1_000
    cleaned = "".join(ch for ch in raw if ch.isdigit() or ch in {".", " ", "-", "+"}).replace(" ", "").strip(".")
    if not cleaned:
        return None
    try:
        if "." in cleaned:
            return int(float(cleaned) * multiplier)
        return int(cleaned) * multiplier
    except ValueError:
        return None


def parse_percent_value(text):
    if text is None:
        return None
    raw = str(text).strip().replace(",", ".").replace("%", "").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def parse_last_seen_to_minutes(text):
    if not text:
        return None
    t = text.strip().lower().replace("\u00a0", " ")
    m = re.search(r"(\d+)", t)
    if not m:
        return None
    value = int(m.group(1))
    if "мин" in t or "minute" in t:
        return value
    if "час" in t or "ч." in t or "hour" in t:
        return value * 60
    if "дн" in t or "день" in t or "дня" in t or "day" in t:
        return value * 60 * 24
    if "мес" in t or "month" in t:
        return value * 60 * 24 * 30
    return None


def detect_language_hint(title, description):
    text = f"{title} {description}".lower()
    rules = [
        ("english", ["english", "английск", "ielts", "toefl"]),
        ("russian", ["русский", "грамот", "правопис", "редактор", "словар"]),
        ("german", ["немецк", "deutsch", "german"]),
        ("spanish", ["испан", "español", "espanol", "castellano"]),
        ("french", ["француз", "francais", "français"]),
        ("korean", ["корейск", "korean", "topik"]),
        ("japanese", ["япон", "nihongo"]),
        ("chinese", ["китайск", "chinese", "hsk"]),
        ("arabic", ["арабск", "arabic", "tajwid", "таджвид"]),
        ("italian", ["итальян", "italiano"]),
        ("turkish", ["турецк", "turkish"]),
        ("hebrew", ["иврит", "hebrew"]),
        ("czech", ["чешск", "czech"]),
    ]
    for label, keywords in rules:
        if any(k in text for k in keywords):
            return label
    return "unknown"


def build_text_description(title, description):
    parts = []
    if title:
        parts.append(f"title: {title}")
    if description:
        parts.append(f"description: {description}")
    return " | ".join(parts)


def build_extended_stats(title, description, last_seen_text, avg_reach, subscribers):
    description = description or ""
    title = title or ""
    engagement_ratio = None
    if avg_reach and subscribers and subscribers > 0:
        engagement_ratio = round(avg_reach / subscribers, 6)
    return {
        "title_length": len(title),
        "description_length": len(description),
        "word_count": len(re.findall(r"\w+", description, flags=re.UNICODE)),
        "has_contact": int(("@" in description) or ("@" in title)),
        "has_link": int(("http://" in description) or ("https://" in description) or ("t.me/" in description)),
        "has_ad_label": int(("реклама" in description.lower()) or ("ads" in description.lower())),
        "has_18plus": int(("18+" in description.lower()) or ("18+" in title.lower())),
        "last_seen_minutes": parse_last_seen_to_minutes(last_seen_text),
        "language_hint": detect_language_hint(title, description),
        "engagement_ratio": engagement_ratio,
    }


def extract_username_from_url(url):
    parsed = urlparse(url or "")
    path = parsed.path.strip("/")
    if "channel/" in path:
        return path.split("channel/", 1)[-1].lstrip("@").strip("/")
    return ""


def make_stat_url(channel_url):
    url = (channel_url or "").rstrip("/")
    if url.endswith("/stat"):
        return url
    return url + "/stat"


def extract_first(patterns, text, flags=re.IGNORECASE):
    for pattern in patterns:
        m = re.search(pattern, text, flags)
        if m:
            return m
    return None


def parse_geo_category(text):
    country = None
    category = None

    m_geo = re.search(r"Гео и язык канала\s+([^\n]+)", text, flags=re.IGNORECASE)
    if m_geo:
        payload = m_geo.group(1).strip()
        parts = [x.strip() for x in re.split(r"[,/|]", payload) if x.strip()]
        if parts:
            country = parts[0]

    m_cat = re.search(r"Категория\s+([^\n]+)", text, flags=re.IGNORECASE)
    if m_cat:
        category = m_cat.group(1).strip()

    return country, category


def extract_json_candidates(html):
    candidates = []
    patterns = [
        r'__NEXT_DATA__"\s*[^>]*>(.*?)</script>',
        r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>',
        r'window\.__INITIAL_STATE__\s*=\s*({.*?});',
        r'window\.__INITIAL_DATA__\s*=\s*({.*?});',
        r'window\.__NUXT__\s*=\s*({.*?});',
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, html, flags=re.DOTALL | re.IGNORECASE):
            payload = m.group(1).strip()
            if payload:
                candidates.append(payload)
    return candidates


def walk_json(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from walk_json(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk_json(item)


def find_number_in_json_candidates(html, key_hints):
    for payload in extract_json_candidates(html):
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        for k, v in walk_json(obj):
            if not isinstance(k, str):
                continue
            low = k.lower()
            if any(h in low for h in key_hints):
                if isinstance(v, (int, float)):
                    return v
                parsed = safe_int_from_text(v)
                if parsed is not None:
                    return parsed
    return None


def find_string_in_json_candidates(html, key_hints):
    for payload in extract_json_candidates(html):
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        for k, v in walk_json(obj):
            if not isinstance(k, str):
                continue
            low = k.lower()
            if any(h in low for h in key_hints) and isinstance(v, str) and v.strip():
                return v.strip()
    return None


def compute_derived_metrics(updated):
    subscribers = safe_int_from_text(updated.get("subscribers"))
    avg_reach = safe_int_from_text(updated.get("avg_reach"))
    avg_ad_reach = safe_int_from_text(updated.get("avg_ad_reach_1_post"))
    stories_avg = safe_int_from_text(updated.get("stories_avg_reach"))
    mentions_total = safe_int_from_text(updated.get("mentions_total"))
    reposts_total = safe_int_from_text(updated.get("reposts_total"))
    posts_month = safe_int_from_text(updated.get("posts_month"))

    updated["avg_reach_to_subscribers_ratio"] = round(avg_reach / subscribers, 6) if avg_reach and subscribers else None
    updated["ad_reach_to_subscribers_ratio"] = round(avg_ad_reach / subscribers, 6) if avg_ad_reach and subscribers else None
    updated["stories_reach_to_subscribers_ratio"] = round(stories_avg / subscribers, 6) if stories_avg and subscribers else None
    updated["mentions_per_1000_subscribers"] = round(mentions_total / subscribers * 1000, 6) if mentions_total and subscribers else None
    updated["reposts_per_1000_subscribers"] = round(reposts_total / subscribers * 1000, 6) if reposts_total and subscribers else None
    updated["posts_per_week_estimated"] = round(posts_month / 4.345, 6) if posts_month else None
    return updated


def parse_stat_page_extended(html, stat_url, row):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"[ \t]+", " ", text)

    title_tag = soup.find("title")
    h1 = soup.find("h1")
    meta_desc = soup.find("meta", attrs={"name": "description"})

    title = row.get("title", "")
    if h1 and h1.get_text(" ", strip=True):
        title = h1.get_text(" ", strip=True)
    elif title_tag and title_tag.get_text(" ", strip=True):
        title = title_tag.get_text(" ", strip=True)
    else:
        title = find_string_in_json_candidates(html, ["title", "channel_title", "name"]) or title

    description = row.get("description", "")
    if meta_desc and meta_desc.get("content"):
        description = meta_desc.get("content").strip()
    else:
        description = find_string_in_json_candidates(html, ["description", "about"]) or description

    country, category = parse_geo_category(text)

    updated = {k: row.get(k) for k in ALL_FIELDS}
    updated["source_page"] = stat_url
    updated["source_page_type"] = "channel_stat"
    updated["channel_url"] = row.get("channel_url", "")
    updated["username"] = row.get("username", "") or extract_username_from_url(row.get("channel_url", ""))
    updated["title"] = title
    updated["description"] = description
    updated["country"] = country or row.get("country")
    updated["category"] = category or row.get("category")
    updated["tag_group"] = row.get("tag_group")
    updated["rank_subscribers"] = row.get("rank_subscribers")
    updated["last_seen_text"] = row.get("last_seen_text")

    subscribers_json = find_number_in_json_candidates(html, ["subscribers", "members", "followers"])
    avg_reach_json = find_number_in_json_candidates(html, ["avg_reach", "average_reach", "reach_avg"])
    citation_json = find_number_in_json_candidates(html, ["citation_index", "citability", "ci"])

    m = extract_first([
        r"(\d[\d\s]*)\s*[-\n\r ]*\s*([+-]?\d[\d\s]*)\s*сегодня\s*([+-]?\d[\d\s]*)\s*за неделю\s*([+-]?\d[\d\s]*)\s*за месяц\s*([0-9A-Za-zА-Яа-я .]+)\s*(\d[\d\s]*)\s*ПОДПИСЧИКИ",
    ], text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        updated["subscribers"] = safe_int_from_text(m.group(1))
        updated["subscribers_today_delta"] = safe_int_from_text(m.group(2))
        updated["subscribers_week_delta"] = safe_int_from_text(m.group(3))
        updated["subscribers_month_delta"] = safe_int_from_text(m.group(4))
        updated["subscribers_baseline_date"] = m.group(5).strip()
        updated["subscribers_baseline_value"] = safe_int_from_text(m.group(6))
    else:
        updated["subscribers"] = subscribers_json or safe_int_from_text(row.get("subscribers"))

    m = extract_first([
        r"([+-]?\d[\d\s]*)\s*[-\n\r ]*\s*([+-]?\d[\d\s]*)\s*подписки\s*([+-]?\d[\d\s]*)\s*отписки\s*ПОДПИСКИ/ОТПИСКИ\s*ЗА 24 ЧАСА"
    ], text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        updated["net_subscriptions_24h"] = safe_int_from_text(m.group(1))
        updated["subscriptions_24h"] = safe_int_from_text(m.group(2))
        updated["unsubscriptions_24h"] = safe_int_from_text(m.group(3))

    m = extract_first([
        r"(\d[\d\s]*\.?\d*)\s*[-\n\r ]*\s*(\d[\d\s]*)\s*уп\.\s*каналов\s*(\d[\d\s]*)\s*упоминаний\s*(\d[\d\s]*)\s*репостов\s*([0-9A-Za-zА-Яа-я .]+)\s*(\d[\d\s]*\.?\d*)\s*ИНДЕКС ЦИТИРОВАНИЯ"
    ], text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        updated["citation_index"] = safe_int_from_text(m.group(1))
        updated["mentions_channels"] = safe_int_from_text(m.group(2))
        updated["mentions_total"] = safe_int_from_text(m.group(3))
        updated["reposts_total"] = safe_int_from_text(m.group(4))
        updated["citation_index_date"] = m.group(5).strip()
        updated["citation_index_baseline_value"] = safe_int_from_text(m.group(6))
    else:
        updated["citation_index"] = citation_json or safe_int_from_text(row.get("citation_index"))

    m = extract_first([
        r"(\d[\d\s]*)\s*[-\n\r ]*\s*([\d.,]+%)\s*ERR\s*([\d.,]+%)\s*ERR24\s*([0-9A-Za-zА-Яа-я .]+)\s*(\d[\d\s]*)\s*СРЕДНИЙ ОХВАТ\s*1 ПУБЛИКАЦИИ"
    ], text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        updated["avg_reach"] = safe_int_from_text(m.group(1))
        updated["avg_reach_1_post"] = safe_int_from_text(m.group(1))
        updated["err_percent"] = parse_percent_value(m.group(2))
        updated["err24_percent"] = parse_percent_value(m.group(3))
        updated["avg_reach_date"] = m.group(4).strip()
        updated["avg_reach_baseline_value"] = safe_int_from_text(m.group(5))
    else:
        updated["avg_reach"] = avg_reach_json or safe_int_from_text(row.get("avg_reach"))

    m = extract_first([
        r"(\d[\d\s.,kKmM]+)\s*[-\n\r ]*\s*(\d[\d\s.,kKmM]+)\s*за 12 часов\s*(\d[\d\s.,kKmM]+)\s*за 24 часа\s*(\d[\d\s.,kKmM]+)\s*за 48 часов\s*СРЕДНИЙ РЕКЛАМНЫЙ\s*ОХВАТ 1 ПУБЛИКАЦИИ"
    ], text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        updated["avg_ad_reach_1_post"] = safe_int_from_text(m.group(1))
        updated["avg_ad_reach_12h"] = safe_int_from_text(m.group(2))
        updated["avg_ad_reach_24h"] = safe_int_from_text(m.group(3))
        updated["avg_ad_reach_48h"] = safe_int_from_text(m.group(4))

    m = extract_first([
        r"(\d[\d\s.,kKmM]+)\s*[-\n\r ]*\s*(\d[\d\s.,kKmM]+)\s*за 12 часов\s*(\d[\d\s.,kKmM]+)\s*за 24 часа\s*(\d[\d\s.,kKmM]+)\s*за 48 часов.*?(\d[\d\s]*)\s*бусты.*?(\d[\d\s]*)\s*реакции.*?(\d[\d\s]*)\s*пересылки.*?STORIES"
    ], text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        updated["stories_avg_reach"] = safe_int_from_text(m.group(1))
        updated["stories_reach_12h"] = safe_int_from_text(m.group(2))
        updated["stories_reach_24h"] = safe_int_from_text(m.group(3))
        updated["stories_reach_48h"] = safe_int_from_text(m.group(4))
        updated["stories_boosts"] = safe_int_from_text(m.group(5))
        updated["stories_reactions"] = safe_int_from_text(m.group(6))
        updated["stories_forwards"] = safe_int_from_text(m.group(7))

    m = extract_first([
        r"([0-9A-Za-zА-Яа-я ]+)\s*[-\n\r ]*\s*(\d{2}\.\d{2}\.\d{4})\s*канал создан\s*(\d{2}\.\d{2}\.\d{4})\s*добавлен в TGStat\s*ВОЗРАСТ КАНАЛА"
    ], text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        updated["channel_age_text"] = m.group(1).strip()
        updated["channel_created_date"] = m.group(2)
        updated["added_to_tgstat_date"] = m.group(3)

    m = extract_first([
        r"(\d[\d\s]*)\s*всего\s*[-\n\r ]*\s*(\d[\d\s]*)\s*вчера\s*(\d[\d\s]*)\s*за неделю\s*(\d[\d\s]*)\s*за месяц\s*ПУБЛИКАЦИИ"
    ], text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        updated["posts_total"] = safe_int_from_text(m.group(1))
        updated["posts_yesterday"] = safe_int_from_text(m.group(2))
        updated["posts_week"] = safe_int_from_text(m.group(3))
        updated["posts_month"] = safe_int_from_text(m.group(4))

    m = extract_first([r"(\d[\d.,]*)%\s*подписчиков читают посты канала"], text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        updated["readers_percent"] = parse_percent_value(m.group(1))

    m = extract_first([r"(\d[\d.,]*)%\s*читают посты\s*в первые 24 часа"], text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        updated["readers_24h_percent"] = parse_percent_value(m.group(1))

    m = extract_first([
        r"([\d.,]+%)\s*[-\n\r ]*\s*(\d[\d\s]*)\s*пересылки\s*(\d[\d\s]*)\s*комментарии\s*(\d[\d\s]*)\s*реакции\s*([0-9A-Za-zА-Яа-я .]+)\s*([\d.,]+)\s*ВОВЛЕЧЕННОСТЬ\s*ПОДПИСЧИКОВ\s*\(ER\)"
    ], text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        updated["er_percent"] = parse_percent_value(m.group(1))
        updated["er_forward_count"] = safe_int_from_text(m.group(2))
        updated["er_comment_count"] = safe_int_from_text(m.group(3))
        updated["er_reaction_count"] = safe_int_from_text(m.group(4))
        updated["er_date"] = m.group(5).strip()
        updated["er_baseline_value"] = parse_percent_value(m.group(6))

    m = extract_first([r"([\d.,]+%)\s*мужчины\s*([\d.,]+%)\s*женщины"], text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        updated["male_percent"] = parse_percent_value(m.group(1))
        updated["female_percent"] = parse_percent_value(m.group(2))

    ext = build_extended_stats(
        updated.get("title", ""),
        updated.get("description", ""),
        updated.get("last_seen_text"),
        updated.get("avg_reach"),
        updated.get("subscribers"),
    )
    updated.update(ext)
    updated["text_description"] = build_text_description(updated.get("title", ""), updated.get("description", ""))
    updated = compute_derived_metrics(updated)
    return updated


def scrape_page(page, url, request_delay):
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        humanize_page(page)
        time.sleep(request_delay)
        html = page.content()
        if is_cloudflare_challenge_html(html):
            return html, "cloudflare_challenge"
        return html, None
    except PlaywrightTimeoutError:
        return "", "timeout"
    except Exception as e:
        return "", repr(e)


def load_state(path=STATE_JSON):
    p = Path(path)
    if not p.exists():
        return {"processed_urls": [], "last_input_index": -1, "written_rows": 0}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(processed_urls, last_input_index, written_rows):
    atomic_write_json(
        STATE_JSON,
        {
            "processed_urls": sorted(processed_urls),
            "last_input_index": last_input_index,
            "written_rows": written_rows,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_existing_output_urls(path):
    p = Path(path)
    if not p.exists():
        return set()
    processed = set()
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = normalize_url(row.get("channel_url", ""))
            if url:
                processed.add(url)
    return processed


class AppendCsvWriter:
    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.fieldnames = fieldnames
        self.file = None
        self.writer = None
        self._open()

    def _open(self):
        file_exists = self.path.exists()
        self.file = self.path.open("a", newline="", encoding="utf-8-sig")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if not file_exists or self.path.stat().st_size == 0:
            self.writer.writeheader()
            self.file.flush()
            os.fsync(self.file.fileno())

    def write_row(self, row):
        safe_row = {k: row.get(k) for k in self.fieldnames}
        self.writer.writerow(safe_row)
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        if self.file:
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()


class AppendJsonlWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open("a", encoding="utf-8")

    def write_row(self, row):
        self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        if self.file:
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()


def main():
    load_env()
    request_delay = load_float_from_env("REQUEST_DELAY", REQUEST_DELAY)
    headless = load_bool_from_env("PLAYWRIGHT_HEADLESS", False)
    manual_challenge = load_bool_from_env("PLAYWRIGHT_MANUAL_CHALLENGE", True)
    postprocess_limit = load_int_from_env("POSTPROCESS_LIMIT", 0)

    rows = read_csv_rows(INPUT_CSV)
    if postprocess_limit > 0:
        rows = rows[:postprocess_limit]
    if not rows:
        debug_log("input csv is empty")
        return

    state = load_state(STATE_JSON)
    processed_urls = set(state.get("processed_urls", []))
    processed_urls_from_csv = read_existing_output_urls(OUTPUT_CSV)
    processed_urls |= processed_urls_from_csv

    debug_log(f"resume: already processed urls={len(processed_urls)}")
    debug_log(f"resume: state last_input_index={state.get('last_input_index', -1)}")

    all_headers = load_headers_from_env_path()
    user_agent, extra_headers = split_user_agent_from_headers(all_headers)
    cookies = load_cookies_from_env_path()
    proxy = load_proxy_from_env()

    csv_writer = AppendCsvWriter(OUTPUT_CSV, ALL_FIELDS)
    jsonl_writer = AppendJsonlWriter(OUTPUT_JSONL)

    written_rows = len(processed_urls_from_csv)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=headless,
                channel="chrome",
                proxy=proxy,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-first-run",
                    "--disable-features=IsolateOrigins,site-per-process",
                ],
            )
            context = browser.new_context(
                user_agent=user_agent,
                locale="ru-RU",
                viewport={"width": 1440, "height": 900},
                extra_http_headers=extra_headers,
            )
            if cookies:
                context.add_cookies(cookies)
            page = context.new_page()

            warmup_url = "https://tgstat.ru/"
            debug_log(f"warmup: open {warmup_url}")
            page.goto(warmup_url, wait_until="domcontentloaded", timeout=60000)
            humanize_page(page)
            html = page.content()

            if manual_challenge and is_cloudflare_challenge_html(html):
                print("\nCloudflare challenge detected.")
                print("Пройди проверку вручную в открывшемся окне браузера, затем нажми Enter здесь в консоли...\n")
                input()
                humanize_page(page)
                html = page.content()

            if is_cloudflare_challenge_html(html):
                debug_log("warmup: Cloudflare challenge still active, abort")
                context.storage_state(path=OUTPUT_STORAGE_STATE)
                browser.close()
                return

            context.storage_state(path=OUTPUT_STORAGE_STATE)
            debug_log(f"warmup: saved browser state={OUTPUT_STORAGE_STATE}")

            for idx, row in enumerate(rows):
                channel_url = normalize_url(row.get("channel_url", ""))
                if not channel_url:
                    log_error(f"row {idx}: empty channel_url")
                    continue

                if channel_url in processed_urls:
                    continue

                stat_url = make_stat_url(channel_url)
                debug_log(f"[{idx + 1}/{len(rows)}] postprocess {stat_url}")

                html, error = scrape_page(page, stat_url, request_delay)

                if error == "cloudflare_challenge" and manual_challenge:
                    print(f"\nCloudflare challenge detected on {stat_url}")
                    print("Пройди проверку вручную в открывшемся окне браузера, затем нажми Enter...\n")
                    input()
                    humanize_page(page)
                    html = page.content()
                    error = "cloudflare_challenge" if is_cloudflare_challenge_html(html) else None

                if error:
                    log_error(f"{stat_url} -> {error}")
                    save_state(processed_urls, idx - 1, written_rows)
                    continue

                try:
                    updated = parse_stat_page_extended(html, stat_url, row)
                    csv_writer.write_row(updated)
                    jsonl_writer.write_row(updated)
                    processed_urls.add(channel_url)
                    written_rows += 1
                    save_state(processed_urls, idx, written_rows)
                    debug_log(
                        f"saved: {channel_url} | subs={updated.get('subscribers')} "
                        f"| reach={updated.get('avg_reach')} | ci={updated.get('citation_index')}"
                    )
                except Exception as e:
                    log_error(f"{stat_url} parse_error -> {repr(e)}")
                    save_state(processed_urls, idx - 1, written_rows)

            context.storage_state(path=OUTPUT_STORAGE_STATE)
            browser.close()

    finally:
        csv_writer.close()
        jsonl_writer.close()

    debug_log("run: done")
    debug_log(f"run: processed urls total={len(processed_urls)}")
    debug_log(f"run: written rows total={written_rows}")
    debug_log(f"run: output csv={OUTPUT_CSV}")
    debug_log(f"run: output jsonl={OUTPUT_JSONL}")
    debug_log(f"run: state json={STATE_JSON}")
    debug_log(f"run: browser state={OUTPUT_STORAGE_STATE}")


if __name__ == "__main__":
    main()