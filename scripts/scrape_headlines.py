from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def clean_headline(text: str) -> str:
    return " ".join(str(text).split()).strip()


def extract_headline(url: str, html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")

    domain_specific = []
    if "foxnews.com" in url:
        domain_specific = [
            ("h1", {"class": "headline speakable"}),
            ("h1", {"class": "headline"}),
        ]
    elif "nbcnews.com" in url:
        domain_specific = [
            ("h1", {"class": "article-hero-headline__htag"}),
            ("h1", {"class": "headline"}),
        ]

    for tag, attrs in domain_specific:
        match = soup.find(tag, attrs=attrs)
        if match and match.get_text(strip=True):
            return clean_headline(match.get_text())

    for attrs in [
        {"property": "og:title"},
        {"name": "twitter:title"},
    ]:
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            return clean_headline(tag["content"])

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return clean_headline(h1.get_text())

    return None


def scrape_url(url: str, timeout: int = 15) -> Optional[str]:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return extract_headline(url, response.text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape headlines from Fox/NBC article URLs.")
    parser.add_argument("--input", required=True, help="CSV with a 'url' column.")
    parser.add_argument("--output", required=True, help="Destination CSV path.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between requests in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    if "url" not in df.columns:
        raise ValueError("Input CSV must contain a 'url' column.")

    rows = []
    for index, url in enumerate(df["url"].astype(str).tolist(), start=1):
        try:
            headline = scrape_url(url)
            status = "ok" if headline else "missing"
        except Exception as exc:
            headline = None
            status = f"error: {exc}"

        rows.append({"url": url, "headline": headline, "status": status})
        print(f"[{index}/{len(df)}] {status} :: {url}")
        time.sleep(args.sleep)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"saved scraped headlines to {output_path}")


if __name__ == "__main__":
    main()
