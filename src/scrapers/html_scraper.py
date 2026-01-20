"""
HTML Scraper Module

This module provides functionality for extracting structured data from HTML content.
"""

from .base_scraper import BaseScraper
from .mixins import HTMLExtractorMixin


class HTMLScraper(BaseScraper, HTMLExtractorMixin):
    """
    HTML Scraper Implementation.

    This scraper extracts structured data from HTML content.
    Note that fetching HTML content is handled by PlaywrightScraper.
    """

    async def fetch_content(self, url: str, proxy: str | None = None) -> str:
        """
        Fetch HTML content from a given URL.

        Note: This method is not implemented in this class as HTML content
        fetching is handled by PlaywrightScraper.

        Raises:
            NotImplementedError: Always raised as this method is not implemented here
        """
        raise NotImplementedError("HTML content is fetched by PlaywrightScraper")

    async def extract(self, content: str) -> dict:
        """
        Extract structured data from HTML content.

        Args:
            content: Raw HTML content to extract data from

        Returns:
            Dictionary containing title, text, and links
        """
        return self.extract_html_data(content)
