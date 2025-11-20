"""
CyberScraper-2077 Scrapers Module

This module provides various scraper implementations for different content types:
- PlaywrightScraper: Advanced web scraper with anti-detection features
- HTMLScraper: HTML content parser using BeautifulSoup
- JSONScraper: JSON content parser
"""

from .playwright_scraper import PlaywrightScraper
from .html_scraper import HTMLScraper
from .json_scraper import JSONScraper