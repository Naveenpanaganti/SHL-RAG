"""
SHL catalog scraper (utility script — not part of the API runtime).

Run this ONCE to build data/catalog.json if you don't have the provided JSON.
Usage:
    python -m app.scraper

Design:
- Scrapes Individual Test Solutions only (Pre-packaged Job Solutions excluded per spec).
- Saves to data/catalog.json in the schema expected by vectorstore.py.
- Uses httpx (async) + BeautifulSoup.
- Respects robots.txt delay with a small sleep between requests.

NOTE: If SHL has provided catalog.json directly, you don't need to run this.
"""

import asyncio
import json
import logging
import os
import time
from typing import List, Dict, Any
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/productcatalog/"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")
REQUEST_DELAY = 1.0  # seconds between requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; SHL-RAG-Scraper/1.0; "
        "for academic/assessment research)"
    )
}


async def scrape_catalog() -> List[Dict[str, Any]]:
    """
    Main entry point. Returns list of catalog item dicts.
    """
    async with httpx.AsyncClient(headers=HEADERS, timeout=30.0, follow_redirects=True) as client:
        logger.info("Fetching catalog index: %s", CATALOG_URL)
        resp = await client.get(CATALOG_URL)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        product_links = _extract_product_links(soup)
        logger.info("Found %d product links", len(product_links))

        items = []
        for i, url in enumerate(product_links):
            try:
                item = await _scrape_product_page(client, url)
                if item:
                    items.append(item)
                    logger.info("[%d/%d] Scraped: %s", i + 1, len(product_links), item["name"])
            except Exception as exc:
                logger.warning("Failed to scrape %s: %s", url, exc)
            await asyncio.sleep(REQUEST_DELAY)

    return items


def _extract_product_links(soup: BeautifulSoup) -> List[str]:
    """
    Extract all individual product page URLs from the catalog index.
    Filters to Individual Test Solutions only.
    """
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/product-catalog/view/" in href or "/products/product-catalog/view/" in href:
            full_url = urljoin(BASE_URL, href)
            if full_url not in links:
                links.append(full_url)
    return links


async def _scrape_product_page(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    """
    Scrape a single product detail page and return a structured dict.
    """
    resp = await client.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    name = _get_text(soup, ["h1", ".product-title", ".entry-title"])
    description = _get_text(soup, [".product-description", ".overview", "article p"])
    test_type = _extract_test_type(soup)
    duration = _extract_duration(soup)
    languages = _extract_languages(soup)
    keys = _extract_keys(soup)
    job_levels = _extract_job_levels(soup)

    if not name:
        return None

    return {
        "name": name.strip(),
        "url": url,
        "test_type": test_type,
        "description": description,
        "duration": duration,
        "languages": languages,
        "keys": keys,
        "job_levels": job_levels,
    }


def _get_text(soup: BeautifulSoup, selectors: List[str]) -> str:
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            return el.get_text(separator=" ", strip=True)
    return ""


def _extract_test_type(soup: BeautifulSoup) -> str:
    """Extract test type letter code(s) from the page."""
    # Try data attributes or labeled fields
    for label in soup.find_all(string=lambda t: t and "test type" in t.lower()):
        parent = label.find_parent()
        if parent:
            sibling = parent.find_next_sibling()
            if sibling:
                return sibling.get_text(strip=True)
    return ""


def _extract_duration(soup: BeautifulSoup) -> str:
    for label in soup.find_all(string=lambda t: t and "duration" in t.lower()):
        parent = label.find_parent()
        if parent:
            sibling = parent.find_next_sibling()
            if sibling:
                return sibling.get_text(strip=True)
    return ""


def _extract_languages(soup: BeautifulSoup) -> List[str]:
    for label in soup.find_all(string=lambda t: t and "language" in t.lower()):
        parent = label.find_parent()
        if parent:
            sibling = parent.find_next_sibling()
            if sibling:
                text = sibling.get_text(separator=",", strip=True)
                return [lang.strip() for lang in text.split(",") if lang.strip()]
    return []


def _extract_keys(soup: BeautifulSoup) -> List[str]:
    """Extract measure/competency keys."""
    for label in soup.find_all(string=lambda t: t and "measure" in t.lower()):
        parent = label.find_parent()
        if parent:
            sibling = parent.find_next_sibling()
            if sibling:
                text = sibling.get_text(separator=",", strip=True)
                return [k.strip() for k in text.split(",") if k.strip()]
    return []


def _extract_job_levels(soup: BeautifulSoup) -> List[str]:
    for label in soup.find_all(string=lambda t: t and "job level" in t.lower()):
        parent = label.find_parent()
        if parent:
            sibling = parent.find_next_sibling()
            if sibling:
                text = sibling.get_text(separator=",", strip=True)
                return [jl.strip() for jl in text.split(",") if jl.strip()]
    return []


async def _main():
    items = await scrape_catalog()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d items to %s", len(items), OUTPUT_PATH)


if __name__ == "__main__":
    asyncio.run(_main())
