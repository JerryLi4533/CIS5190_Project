from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import html
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "raw" / "recent_validation.csv"
DEFAULT_CUTOFF = "2026-02-01"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

FOX_FEEDS = [
    "https://moxie.foxnews.com/google-publisher/latest.xml",
    "https://moxie.foxnews.com/google-publisher/politics.xml",
    "https://moxie.foxnews.com/google-publisher/us.xml",
    "https://moxie.foxnews.com/google-publisher/world.xml",
    "https://moxie.foxnews.com/google-publisher/entertainment.xml",
    "https://moxie.foxnews.com/google-publisher/health.xml",
    "https://moxie.foxnews.com/google-publisher/lifestyle.xml",
    "https://moxie.foxnews.com/google-publisher/sports.xml",
    "https://moxie.foxnews.com/google-publisher/tech.xml",
]

NBC_FEEDS = [
    "https://feeds.nbcnews.com/nbcnews/public/news",
    "https://feeds.nbcnews.com/nbcnews/public/politics",
    "https://feeds.nbcnews.com/nbcnews/public/world",
    "https://feeds.nbcnews.com/nbcnews/public/us-news",
    "https://feeds.nbcnews.com/nbcnews/public/business",
    "https://feeds.nbcnews.com/nbcnews/public/health",
    "https://feeds.nbcnews.com/nbcnews/public/science",
    "https://feeds.nbcnews.com/nbcnews/public/tech",
    "https://feeds.nbcnews.com/nbcnews/public/pop-culture",
]


def clean_text(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def child_text(item: ET.Element, name: str) -> str:
    child = item.find(name)
    if child is not None and child.text:
        return child.text
    return ""


def fetch_feed(feed_url: str, timeout: int) -> list[dict[str, str]]:
    response = requests.get(feed_url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    rows = []
    for item in root.findall(".//item"):
        title = clean_text(child_text(item, "title"))
        link = clean_text(child_text(item, "link"))
        pub_date = clean_text(child_text(item, "pubDate"))
        if title and link:
            rows.append(
                {
                    "url": link,
                    "headline": title,
                    "published": pub_date,
                    "feed_url": feed_url,
                }
            )
    return rows


def source_for_url(url: str) -> str:
    url = url.lower()
    if "foxnews.com" in url:
        return "FoxNews"
    if "nbcnews.com" in url:
        return "NBC"
    return "Unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape post-January-2026 Fox/NBC RSS headlines for pseudo-hidden validation."
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF, help="Keep articles on/after this YYYY-MM-DD date.")
    parser.add_argument("--max-per-source", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cutoff = datetime.fromisoformat(args.cutoff).replace(tzinfo=timezone.utc)
    feeds = FOX_FEEDS + NBC_FEEDS
    all_rows: list[dict[str, str]] = []

    for index, feed_url in enumerate(feeds, start=1):
        try:
            rows = fetch_feed(feed_url, timeout=args.timeout)
            print(f"[{index}/{len(feeds)}] {len(rows):3d} rows :: {feed_url}")
            all_rows.extend(rows)
        except Exception as exc:
            print(f"[{index}/{len(feeds)}] error: {exc} :: {feed_url}")
        time.sleep(args.sleep)

    kept: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    seen_headlines: set[tuple[str, str]] = set()
    source_counts = {"FoxNews": 0, "NBC": 0}

    for row in all_rows:
        source = source_for_url(row["url"])
        published_at = parse_date(row["published"])
        if source not in source_counts or published_at is None or published_at < cutoff:
            continue

        key = (source, clean_text(row["headline"]).lower())
        if row["url"] in seen_urls or key in seen_headlines:
            continue
        if source_counts[source] >= args.max_per_source:
            continue

        seen_urls.add(row["url"])
        seen_headlines.add(key)
        source_counts[source] += 1
        kept.append(
            {
                "url": row["url"],
                "headline": row["headline"],
                "source": source,
                "published": published_at.isoformat(),
                "feed_url": row["feed_url"],
            }
        )

    kept.sort(key=lambda row: (row["source"], row["published"], row["headline"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["url", "headline", "source", "published", "feed_url"])
        writer.writeheader()
        writer.writerows(kept)

    print(f"saved {len(kept)} rows to {args.output}")
    print(f"source_counts: {source_counts}")


if __name__ == "__main__":
    main()
