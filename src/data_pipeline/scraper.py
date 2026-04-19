import json
import logging
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class SportsScraper:
    """
    Scrapes sports articles using ESPN's internal JSON API and
    Sports Reference HTML recaps.
    """

    # ESPN hidden JSON API — no HTML parsing needed
    ESPN_API_BASE = "https://site.api.espn.com/apis/site/v2/sports"
    ESPN_CONTENT_API = "https://content.core.api.espn.com/v1/sports/news"

    SPORTS_REF_BASE_URL = "https://www.sports-reference.com"

    # sport -> (espn_path, content_api_sport_path)
    ESPN_SPORT_MAP: Dict[str, str] = {
        "nba":  "basketball/nba",
        "nfl":  "football/nfl",
        "mlb":  "baseball/mlb",
        "nhl":  "hockey/nhl",
    }

    def __init__(self, output_dir: str = "data/raw", delay_seconds: float = 1.0) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay_seconds = delay_seconds
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "GLiNER-Research-Scraper/1.0",
            "Accept": "application/json",
        })
        self.progress_file = self.output_dir / "progress.json"

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def _load_progress(self) -> Dict:
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                return json.load(f)
        return {}

    def _save_progress(self, progress: Dict) -> None:
        with open(self.progress_file, "w") as f:
            json.dump(progress, f, indent=2)

    # ------------------------------------------------------------------
    # HTTP with exponential backoff
    # ------------------------------------------------------------------

    def _get(self, url: str, retries: int = 4) -> Optional[requests.Response]:
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                time.sleep(self.delay_seconds)
                return response
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code
                if status in (429, 403):
                    wait = (2 ** attempt) * self.delay_seconds
                    logger.warning(f"Rate limited ({status}) on {url}. Waiting {wait:.1f}s")
                    time.sleep(wait)
                else:
                    logger.error(f"HTTP {status} on {url}")
                    return None
            except requests.exceptions.RequestException as e:
                wait = (2 ** attempt) * self.delay_seconds
                logger.warning(f"Request failed ({e}). Retry {attempt+1}/{retries} in {wait:.1f}s")
                time.sleep(wait)
        logger.error(f"All retries exhausted for {url}")
        return None

    # ------------------------------------------------------------------
    # ESPN JSON API
    # ------------------------------------------------------------------

    def scrape_espn_articles(
        self, sport: str, max_articles: int = 500
    ) -> List[Dict]:
        """
        Fetch articles from ESPN's internal JSON API for a given sport.

        Args:
            sport: Sport key from ESPN_SPORT_MAP (e.g. 'nba', 'nfl').
            max_articles: Maximum number of articles to collect.

        Returns:
            List of article dicts: id, url, title, description, body,
            sport, source, scraped_at, weak_labels.
        """
        if sport not in self.ESPN_SPORT_MAP:
            raise ValueError(f"Unknown sport '{sport}'. Choose from {list(self.ESPN_SPORT_MAP)}")

        progress = self._load_progress()
        seen_ids = set(progress.get(f"espn_{sport}_completed", []))

        news_url = f"{self.ESPN_API_BASE}/{self.ESPN_SPORT_MAP[sport]}/news?limit={max_articles}"
        logger.info(f"Fetching ESPN {sport.upper()} article list from {news_url}")

        response = self._get(news_url)
        if not response:
            logger.error("Failed to fetch article list")
            return []

        data = response.json()
        raw_articles = data.get("articles", [])
        logger.info(f"Found {len(raw_articles)} articles in API response")

        articles = []
        for raw in raw_articles:
            article_id = str(raw.get("id", ""))

            # Skip already processed
            if article_id in seen_ids:
                continue

            # Skip non-text content types (Media = video clips)
            if raw.get("type") == "Media":
                continue

            article = self._enrich_espn_article(raw, sport)
            if article:
                articles.append(article)
                seen_ids.add(article_id)
                progress[f"espn_{sport}_completed"] = list(seen_ids)
                self._save_progress(progress)

            if len(articles) >= max_articles:
                break

        logger.info(f"Collected {len(articles)} articles for {sport}")
        return articles

    def _enrich_espn_article(self, raw: Dict, sport: str) -> Optional[Dict]:
        """
        Given a raw article object from the ESPN news API, fetch full
        body text from the content API and extract weak supervision labels
        from the categories array.

        Args:
            raw: Single article dict from ESPN news API response.
            sport: Sport key for metadata.

        Returns:
            Enriched article dict or None if body unavailable.
        """
        article_id = str(raw.get("id", ""))
        headline = raw.get("headline", "")
        description = raw.get("description", "")
        web_url = raw.get("links", {}).get("web", {}).get("href", "")

        # Extract weak supervision labels from categories
        # These are pre-tagged entities ESPN already knows about
        weak_labels = self._extract_weak_labels(raw.get("categories", []))

        # Fetch full article body from content API
        body = self._fetch_article_body(article_id, web_url)

        # Fall back to description if body unavailable
        if not body:
            if len(description) > 100:
                body = description
            else:
                logger.warning(f"No body for article {article_id}, skipping")
                return None

        return {
            "id": article_id,
            "url": web_url,
            "title": headline,
            "description": description,
            "body": body,
            "sport": sport,
            "source": "espn_api",
            "scraped_at": datetime.utcnow().isoformat(),
            "weak_labels": weak_labels,  # free entity hints for annotation QA
        }

    def _fetch_article_body(
        self, article_id: str, fallback_url: str
    ) -> Optional[str]:
        """
        Fetch full article body text from ESPN's content API.
        If the API doesn't return body text, fall back to scraping the web URL.

        Args:
            article_id: Unique ID of the article for content API lookup.
            fallback_url: Web URL to scrape if content API fails.

        Returns:
            Cleaned body text or None if unavailable.
        """
        content_url = f"{self.ESPN_CONTENT_API}/{article_id}"
        response = self._get(content_url)

        if response:
            try:
                data = response.json()
                headlines = data.get("headlines", [])
                if headlines:
                    story = headlines[0].get("story", "")
                    if isinstance(story, str) and len(story) > 100:
                        return self.clean_text(story)
            except (ValueError, KeyError):
                pass

        # Fallback: scrape web URL
        if fallback_url:
            logger.debug(f"Content API failed for {article_id}, scraping {fallback_url}")
            return self._scrape_article_html(fallback_url)

        return None

    def _scrape_article_html(self, url: str) -> Optional[str]:
        """
        Fallback HTML scraper for when the content API doesn't return body text.

        Args:
            url: Full ESPN article URL.

        Returns:
            Cleaned body text or None.
        """
        response = self._get(url)
        if not response:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Try multiple ESPN article body class patterns
        body_div = (
            soup.find("div", class_="article-body")
            or soup.find("div", class_="story__body")
            or soup.find("article")
        )

        if not body_div:
            return None

        paragraphs = [p.get_text(strip=True) for p in body_div.find_all("p")]
        body = " ".join(paragraphs)
        return self.clean_text(body) if len(body) > 100 else None

    def _extract_weak_labels(self, categories: List[Dict]) -> Dict[str, List[str]]:
        """
        Extract pre-tagged entity hints from ESPN's categories array.
        These serve as weak supervision for annotation validation.

        ESPN already tells us which players, teams, and leagues appear
        in each article — we can use this to sanity-check GPT annotations.

        Args:
            categories: The 'categories' array from an ESPN article object.

        Returns:
            Dict with keys 'players', 'teams', 'leagues' mapping to name lists.
        """
        weak_labels: Dict[str, List[str]] = {
            "players": [],
            "teams": [],
            "leagues": [],
        }

        for cat in categories:
            cat_type = cat.get("type", "")
            description = cat.get("description", "")

            if cat_type == "athlete" and description:
                weak_labels["players"].append(description)
            elif cat_type == "team" and description:
                weak_labels["teams"].append(description)
            elif cat_type == "league" and description:
                weak_labels["leagues"].append(description)

        return weak_labels

    # ------------------------------------------------------------------
    # Sports Reference (unchanged — HTML only source)
    # ------------------------------------------------------------------

    def scrape_sports_reference_recaps(
        self, sport: str, season: int, max_games: int = 200
    ) -> List[Dict]:
        """
        Scrape game recap/box-score narratives from sports-reference.com.
        """
        sport_paths = {
            "nba": f"/leagues/NBA_{season}_games.html",
            "nfl": f"/years/{season}/games.htm",
            "mlb": f"/leagues/majors/{season}-schedule.shtml",
        }

        if sport not in sport_paths:
            raise ValueError(f"Sports Reference unsupported sport: {sport}")

        schedule_url = self.SPORTS_REF_BASE_URL + sport_paths[sport]
        response = self._get(schedule_url)
        if not response:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        game_links = []

        for a in soup.find_all("a", href=True):
            if "boxscore" in a["href"] or "box-score" in a["href"]:
                game_links.append(self.SPORTS_REF_BASE_URL + a["href"])
            if len(game_links) >= max_games:
                break

        recaps = []
        for url in game_links:
            recap = self._parse_sports_reference_recap(url, sport, season)
            if recap:
                recaps.append(recap)

        return recaps

    def _parse_sports_reference_recap(
        self, url: str, sport: str, season: int
    ) -> Optional[Dict]:
        response = self._get(url)
        if not response:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        notes_div = soup.find("div", id="game_notes") or soup.find("div", class_="game_summary")
        if not notes_div:
            return None

        body = self.clean_text(notes_div.get_text())
        if len(body) < 50:
            return None

        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else f"{sport} game recap"

        return {
            "url": url,
            "title": title,
            "body": body,
            "sport": sport,
            "season": season,
            "source": "sports_reference",
            "scraped_at": datetime.utcnow().isoformat(),
            "weak_labels": {},
        }

    # ------------------------------------------------------------------
    # Cleaning and deduplication
    # ------------------------------------------------------------------

    def clean_text(self, raw_html: str) -> str:
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ")

        import re
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def deduplicate(self, articles: List[Dict]) -> List[Dict]:
        seen_urls = set()
        seen_hashes = set()
        unique = []

        for article in articles:
            url = article.get("url", "")
            body_hash = hashlib.sha256(
                article.get("body", "")[:500].encode()
            ).hexdigest()

            if url in seen_urls or body_hash in seen_hashes:
                continue

            seen_urls.add(url)
            seen_hashes.add(body_hash)
            unique.append(article)

        removed = len(articles) - len(unique)
        logger.info(f"Dedup: {len(articles)} → {len(unique)} ({removed} removed)")
        return unique

    def save_to_jsonl(self, articles: List[Dict], filename: str) -> Path:
        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(articles)} articles to {out_path}")
        return out_path

    def run(self, sports: Optional[List[str]] = None, max_per_source: int = 500) -> None:
        sports = sports or list(self.ESPN_SPORT_MAP.keys())

        for sport in sports:
            logger.info(f"--- Starting {sport.upper()} ---")
            articles = []

            espn_articles = self.scrape_espn_articles(sport, max_per_source)
            articles.extend(espn_articles)

            # Sports Reference disabled — URL patterns need fixing

            # if sport in ("nba", "nfl", "mlb"):
            #     recaps = self.scrape_sports_reference_recaps(sport, 2024, max_per_source // 2)
            #     articles.extend(recaps)

            articles = self.deduplicate(articles)
            self.save_to_jsonl(articles, f"{sport}_raw.jsonl")