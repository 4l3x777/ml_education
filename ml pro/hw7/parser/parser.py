import csv
import json
import os
import re
import time
import uuid
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

BASE_URL = "https://tgstat.ru/"
START_PATH = "language"
REQUEST_DELAY = 0.8
MAX_CHANNEL_PAGES = 1000
LOAD_MORE_CLICKS = 200
STATE_SAVE_EVERY = 10

OUTPUT_CHANNELS_CSV = "tgstat_channels.csv"
OUTPUT_STORAGE_STATE = "storage_state.json"
OUTPUT_CRAWL_STATE = "crawl_state.json"
OUTPUT_ERRORS_LOG = "errors.log"
OUTPUT_DEBUG_LOG = "debug.log"
OUTPUT_DEBUG_INITIAL_HTML = "tgstat_debug_initial.html"
OUTPUT_DEBUG_AFTER_CLICKS_HTML = "tgstat_debug_after_clicks.html"

PLAYWRIGHT_HEADLESS="false"
PLAYWRIGHT_MANUAL_CHALLENGE="true"
TGSTAT_START_PATH="language"

CHANNEL_LINK_PATTERNS = [
    re.compile(r"^/channel/[^/]+/?$"),
    re.compile(r"^/@[^/]+/?$"),
]

LAST_SEEN_MARKERS = ["мин", "час", "дн", "дня", "день", "мес", "сек", "online", "was online"]
SUBSCRIBER_MARKERS = ["подписчик", "subscribers", "subscriber"]


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
        "Referer": BASE_URL,
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


def normalize_url(url):
    if not url:
        return ""
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl()


def is_internal_tgstat_url(url):
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.netloc in {"tgstat.ru", "www.tgstat.ru"}


def classify_url(url):
    path = urlparse(url).path
    if path.startswith("/channel/"):
        return "channel"
    if path.startswith("/language"):
        return "language"
    return "other"


def is_probable_channel_href(href):
    if not href:
        return False
    parsed = urlparse(href)
    if parsed.netloc and parsed.netloc not in {"tgstat.ru", "www.tgstat.ru"}:
        return False
    path = parsed.path or ""
    if any(pattern.match(path) for pattern in CHANNEL_LINK_PATTERNS):
        return True
    return classify_url(href) == "channel"


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
    cleaned = "".join(ch for ch in raw if ch.isdigit() or ch in {".", " "}).replace(" ", "").strip(".")
    if not cleaned:
        return None
    try:
        if "." in cleaned:
            return int(float(cleaned) * multiplier)
        return int(cleaned) * multiplier
    except ValueError:
        return None


def build_text_description(title, description):
    parts = []
    if title:
        parts.append(f"title: {title}")
    if description:
        parts.append(f"description: {description}")
    return " | ".join(parts)


def extract_text_lines(node):
    if node is None:
        return []
    text = node.get_text("\n", strip=True)
    return [line.strip() for line in text.split("\n") if line.strip()]


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
        ("chinese", ["китайск", "chinese"]),
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


def is_show_more_text(text):
    t = (text or "").strip().lower()
    return t in {"показать больше", "показать ещё", "show more", "load more"}


def find_candidate_channel_nodes(soup):
    anchors = soup.find_all("a", href=True)
    candidates = []
    for a in anchors:
        href = normalize_url(urljoin(BASE_URL, a.get("href", "")))
        if not href or not is_internal_tgstat_url(href):
            continue
        if not is_probable_channel_href(href):
            continue
        text = a.get_text(" ", strip=True)
        if is_show_more_text(text):
            continue
        lines = extract_text_lines(a)
        if len(lines) < 1:
            continue
        if len(" ".join(lines)) < 3:
            continue
        candidates.append(a)
    return candidates


def parse_channel_card(node, page_url, rank):
    href = normalize_url(urljoin(page_url, node.get("href", "")))
    lines = extract_text_lines(node)
    if not lines:
        return None

    title = lines[0]
    if not title or len(title) < 2:
        return None

    description = ""
    subscribers = None
    last_seen_text = None

    for i, line in enumerate(lines):
        low = line.lower()
        if any(marker in low for marker in SUBSCRIBER_MARKERS) and i > 0 and subscribers is None:
            subscribers = safe_int_from_text(lines[i - 1])
        if any(marker in low for marker in LAST_SEEN_MARKERS) and last_seen_text is None:
            last_seen_text = line

    if len(lines) > 1:
        for line in lines[1:]:
            low = line.lower()
            if any(marker in low for marker in SUBSCRIBER_MARKERS):
                continue
            if any(marker in low for marker in LAST_SEEN_MARKERS):
                continue
            if safe_int_from_text(line) is not None and len(line) <= 12:
                continue
            description = line
            break

    parsed = urlparse(href)
    username = parsed.path.replace("/channel/", "").lstrip("@").strip("/")
    ext = build_extended_stats(title, description, last_seen_text, None, subscribers)
    return {
        "source_page": page_url,
        "source_page_type": "language",
        "channel_url": href,
        "username": username,
        "title": title,
        "description": description,
        "subscribers": subscribers,
        "avg_reach": None,
        "citation_index": None,
        "country": "Россия",
        "category": "Лингвистика",
        "tag_group": None,
        "rank_subscribers": rank,
        "last_seen_text": last_seen_text,
        "text_description": build_text_description(title, description),
        **ext,
    }


def parse_language_page(html, page_url):
    if is_cloudflare_challenge_html(html):
        debug_log("parse_language_page: detected Cloudflare challenge page")
        return []

    soup = BeautifulSoup(html, "html.parser")
    entries = []
    seen_urls = set()

    candidate_nodes = find_candidate_channel_nodes(soup)
    debug_log(f"parse_language_page: candidate nodes={len(candidate_nodes)}")

    rank = 0
    for node in candidate_nodes:
        entry = parse_channel_card(node, page_url, rank + 1)
        if not entry:
            continue
        href = entry["channel_url"]
        if href in seen_urls:
            continue
        seen_urls.add(href)
        rank += 1
        entry["rank_subscribers"] = rank
        entries.append(entry)

    debug_log(f"parse_language_page: parsed entries={len(entries)}")
    if entries:
        debug_log(f"parse_language_page: first usernames={[e['username'] for e in entries[:10]]}")
    else:
        debug_log("parse_language_page: no entries parsed from HTML")
    return entries


def count_main_catalog_channels_from_html(html, page_url):
    return len(parse_language_page(html, page_url))


def find_metric_near_keyword(text, keyword, window=120):
    lower_text = text.lower()
    lower_keyword = keyword.lower()
    idx = lower_text.find(lower_keyword)
    if idx == -1:
        return None
    snippet = text[max(0, idx - window): idx + window]
    candidates, current = [], []
    for ch in snippet:
        if ch.isdigit() or ch in {" ", ".", ","}:
            current.append(ch)
        else:
            if current:
                candidates.append("".join(current).strip())
                current = []
    if current:
        candidates.append("".join(current).strip())
    for candidate in candidates:
        value = safe_int_from_text(candidate)
        if value is not None:
            return value
    return None


def parse_channel_page(html, page_url, base_entry):
    soup = BeautifulSoup(html, "html.parser")
    entry = dict(base_entry)
    title_tag = soup.find("title")
    page_title = title_tag.get_text(" ", strip=True) if title_tag else ""
    meta_desc = soup.find("meta", attrs={"name": "description"})
    meta_description = meta_desc.get("content", "").strip() if meta_desc and meta_desc.get("content") else ""
    full_text = soup.get_text(" ", strip=True)

    if meta_description:
        entry["description"] = meta_description
    if not entry.get("title") and page_title:
        entry["title"] = page_title
    if entry.get("avg_reach") is None:
        entry["avg_reach"] = (
            find_metric_near_keyword(full_text, "средний охват")
            or find_metric_near_keyword(full_text, "охват")
            or find_metric_near_keyword(full_text, "avg reach")
        )
    if entry.get("citation_index") is None:
        entry["citation_index"] = (
            find_metric_near_keyword(full_text, "индекс цитирования")
            or find_metric_near_keyword(full_text, "цитируемость")
            or find_metric_near_keyword(full_text, "citation")
        )
    if entry.get("subscribers") is None:
        entry["subscribers"] = (
            find_metric_near_keyword(full_text, "подписчики")
            or find_metric_near_keyword(full_text, "subscribers")
        )
    ext = build_extended_stats(
        entry.get("title", ""),
        entry.get("description", ""),
        entry.get("last_seen_text"),
        entry.get("avg_reach"),
        entry.get("subscribers"),
    )
    entry.update(ext)
    entry["text_description"] = build_text_description(entry.get("title", ""), entry.get("description", ""))
    return entry


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


def wait_for_catalog_ready(page, timeout_ms=30000):
    selectors = [
        'a[href*="/channel/"]',
        'a[href^="/@"]',
        'text=Показать больше',
        'text=Показать ещё',
    ]
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        html = page.content()
        if is_cloudflare_challenge_html(html):
            return False
        for selector in selectors:
            try:
                if page.locator(selector).count() > 0:
                    return True
            except Exception:
                pass
        time.sleep(0.5)
    return False


def click_load_more(page, page_url, max_clicks, request_delay):
    html = page.content()
    if is_cloudflare_challenge_html(html):
        debug_log("click_load_more: Cloudflare challenge detected before clicks")
        return html

    previous_count = count_main_catalog_channels_from_html(html, page_url)
    debug_log(f"click_load_more: start count={previous_count}")
    selectors = [
        'text=Показать больше',
        'button:has-text("Показать больше")',
        'a:has-text("Показать больше")',
        'text=Показать ещё',
        'button:has-text("Показать ещё")',
        'a:has-text("Показать ещё")',
    ]
    for click_index in range(max_clicks):
        clicked = False
        for selector in selectors:
            try:
                locator = page.locator(selector).last
                if locator.count() == 0:
                    continue
                locator.scroll_into_view_if_needed(timeout=3000)
                page.wait_for_timeout(500)
                try:
                    locator.click(timeout=6000)
                except Exception:
                    locator.click(timeout=6000, force=True)
                clicked = True
                break
            except Exception:
                continue

        if not clicked:
            debug_log("click_load_more: show more button not found or finished")
            break

        increased = False
        for _ in range(15):
            page.wait_for_timeout(int((request_delay + 0.5) * 1000))
            html = page.content()
            if is_cloudflare_challenge_html(html):
                debug_log("click_load_more: Cloudflare challenge detected after click")
                return html
            current_count = count_main_catalog_channels_from_html(html, page_url)
            if current_count > previous_count:
                debug_log(f"click_load_more: click #{click_index + 1} increased count {previous_count} -> {current_count}")
                previous_count = current_count
                increased = True
                break
        if not increased:
            debug_log(f"click_load_more: click #{click_index + 1} no growth, count={previous_count}")
            break
    return page.content()


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


def load_state(path=OUTPUT_CRAWL_STATE):
    p = Path(path)
    if not p.exists():
        return {"visited": [], "queued": [], "queue": [], "seen_entry_keys": []}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


class SafeCsvWriter:
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
        self.writer.writerow(row)
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        if self.file:
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()


def make_entry_key(entry):
    return entry.get("channel_url", "")


def save_state(visited, queued, queue, seen_entry_keys):
    atomic_write_json(
        OUTPUT_CRAWL_STATE,
        {
            "visited": sorted(visited),
            "queued": sorted(queued),
            "queue": list(queue),
            "seen_entry_keys": sorted(seen_entry_keys),
        },
    )


def main():
    load_env()
    request_delay = load_float_from_env("REQUEST_DELAY", REQUEST_DELAY)
    max_channel_pages = load_int_from_env("MAX_CHANNEL_PAGES", MAX_CHANNEL_PAGES)
    load_more_clicks = load_int_from_env("LOAD_MORE_CLICKS", LOAD_MORE_CLICKS)
    state_save_every = load_int_from_env("STATE_SAVE_EVERY", STATE_SAVE_EVERY)
    headless = load_bool_from_env("PLAYWRIGHT_HEADLESS", False)
    manual_challenge = load_bool_from_env("PLAYWRIGHT_MANUAL_CHALLENGE", True)
    start_path = os.getenv("TGSTAT_START_PATH", START_PATH)
    start_url = urljoin(BASE_URL, start_path)

    all_headers = load_headers_from_env_path()
    user_agent, extra_headers = split_user_agent_from_headers(all_headers)
    cookies = load_cookies_from_env_path()
    proxy = load_proxy_from_env()

    state = load_state(OUTPUT_CRAWL_STATE)
    visited = set(state.get("visited", []))
    queued = set(state.get("queued", []))
    seen_entry_keys = set(state.get("seen_entry_keys", []))
    queue = deque(state["queue"]) if state.get("queue") else deque()

    fieldnames = [
        "source_page", "source_page_type", "channel_url", "username", "title", "description",
        "subscribers", "avg_reach", "citation_index", "country", "category", "tag_group",
        "rank_subscribers", "last_seen_text", "last_seen_minutes", "title_length",
        "description_length", "word_count", "has_contact", "has_link", "has_ad_label",
        "has_18plus", "language_hint", "engagement_ratio", "text_description",
    ]

    writer = SafeCsvWriter(OUTPUT_CHANNELS_CSV, fieldnames)
    try:
        with sync_playwright() as p:
            launch_args = [
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--disable-features=IsolateOrigins,site-per-process",
            ]
            browser = p.chromium.launch(
                headless=headless,
                channel="chrome",
                proxy=proxy,
                args=launch_args,
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

            if not queue:
                debug_log(f"catalog: open {start_url}")
                page.goto(start_url, wait_until="domcontentloaded", timeout=60000)
                humanize_page(page)
                initial_html = page.content()
                Path(OUTPUT_DEBUG_INITIAL_HTML).write_text(initial_html, encoding="utf-8")
                debug_log(f"catalog: saved {OUTPUT_DEBUG_INITIAL_HTML}")

                if manual_challenge and is_cloudflare_challenge_html(initial_html):
                    print("\nCloudflare challenge detected.")
                    print("Пройди проверку вручную в открывшемся окне браузера, затем нажми Enter здесь в консоли...\n")
                    input()
                    humanize_page(page)

                ready = wait_for_catalog_ready(page, timeout_ms=30000)
                time.sleep(request_delay + 2)
                initial_html = page.content()
                Path(OUTPUT_DEBUG_INITIAL_HTML).write_text(initial_html, encoding="utf-8")
                context.storage_state(path=OUTPUT_STORAGE_STATE)
                debug_log(f"catalog: saved browser state={OUTPUT_STORAGE_STATE}")

                if is_cloudflare_challenge_html(initial_html):
                    debug_log("catalog: Cloudflare challenge still active after manual step")
                    browser.close()
                    return

                if not ready:
                    debug_log("catalog: page not fully ready, continuing with current HTML")

                catalog_html = click_load_more(page, start_url, load_more_clicks, request_delay)
                Path(OUTPUT_DEBUG_AFTER_CLICKS_HTML).write_text(catalog_html, encoding="utf-8")
                debug_log(f"catalog: saved {OUTPUT_DEBUG_AFTER_CLICKS_HTML}")

                if is_cloudflare_challenge_html(catalog_html):
                    debug_log("catalog: Cloudflare challenge detected after clicks")
                    context.storage_state(path=OUTPUT_STORAGE_STATE)
                    browser.close()
                    return

                catalog_entries = parse_language_page(catalog_html, start_url)
                debug_log(f"catalog: found entries in rubric={len(catalog_entries)}")
                debug_log(f"catalog: first parsed urls={[e.get('channel_url') for e in catalog_entries[:10]]}")
                for entry in catalog_entries:
                    channel_url = normalize_url(entry["channel_url"])
                    if not channel_url or channel_url in visited or channel_url in queued:
                        continue
                    queue.append(entry)
                    queued.add(channel_url)
                debug_log(f"catalog: queued channels={len(queue)}")
                save_state(visited, queued, queue, seen_entry_keys)

            processed_count = 0
            while queue and processed_count < max_channel_pages:
                current_entry = queue.popleft()
                current_url = normalize_url(current_entry["channel_url"])
                if current_url in visited:
                    continue

                debug_log(f"channel: start #{processed_count + 1} url={current_url}")
                # try:
                #     page.goto(current_url, wait_until="domcontentloaded", timeout=60000)
                #     humanize_page(page)
                #     time.sleep(request_delay)
                #     html = page.content()
                # except PlaywrightTimeoutError:
                #     html = ""
                #     error = "timeout"
                # else:
                #     error = "cloudflare_challenge" if is_cloudflare_challenge_html(html) else None
                error = None

                visited.add(current_url)
                processed_count += 1

                if error:
                    log_error(f"{current_url} -> {error}")
                    debug_log(f"channel: error url={current_url} error={error}")
                    if processed_count % state_save_every == 0 or not queue:
                        save_state(visited, queued, queue, seen_entry_keys)
                    continue

                try:
                    enriched_entry = current_entry #parse_channel_page(html, current_url, current_entry)
                    entry_key = make_entry_key(enriched_entry)
                    if entry_key not in seen_entry_keys:
                        writer.write_row(enriched_entry)
                        seen_entry_keys.add(entry_key)
                        debug_log(
                            f"channel: saved url={current_url} subs={enriched_entry.get('subscribers')} "
                            f"reach={enriched_entry.get('avg_reach')} ci={enriched_entry.get('citation_index')}"
                        )
                except Exception as parse_exc:
                    log_error(f"parse_error {current_url} -> {repr(parse_exc)}")
                    debug_log(f"channel: parse_error url={current_url} error={repr(parse_exc)}")

                if processed_count % state_save_every == 0 or not queue:
                    save_state(visited, queued, queue, seen_entry_keys)

            context.storage_state(path=OUTPUT_STORAGE_STATE)
            browser.close()
    finally:
        writer.close()

    debug_log("run: done")
    debug_log(f"run: visited pages total={len(visited)}")
    debug_log(f"run: unique entries total={len(seen_entry_keys)}")
    debug_log(f"run: saved state={OUTPUT_CRAWL_STATE}")
    debug_log(f"run: saved browser state={OUTPUT_STORAGE_STATE}")
    debug_log(f"run: saved entries csv={OUTPUT_CHANNELS_CSV}")
    debug_log(f"run: saved errors log={OUTPUT_ERRORS_LOG}")
    debug_log(f"run: saved debug log={OUTPUT_DEBUG_LOG}")


if __name__ == "__main__":
    main()