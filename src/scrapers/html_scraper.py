
"""
HTML Scraper Module

This module provides functionality for extracting structured data from HTML content.
It uses BeautifulSoup for parsing HTML documents.
"""

from bs4 import BeautifulSoup
from .base_scraper import BaseScraper
from typing import Dict, Any


class HTMLScraper(BaseScraper):
    """
    HTML Scraper Implementation
    
    This scraper extracts structured data from HTML content using BeautifulSoup.
    Note that fetching HTML content is handled by PlaywrightScraper.
    """
    
    async def fetch_content(self, url: str, proxy: str = None) -> str:
        """
        Fetch HTML content from a given URL.
        
        Note: This method is not implemented in this class as HTML content
        fetching is handled by PlaywrightScraper.
        
        Args:
            url (str): The URL to fetch content from
            proxy (str, optional): Proxy server to use for the request
            
        Raises:
            NotImplementedError: Always raised as this method is not implemented here
        """
        raise NotImplementedError("HTML content is fetched by PlaywrightScraper")

    async def extract(self, content: str) -> Dict[str, Any]:
        """
        Extract structured data from HTML content.
        
        This method parses HTML content and extracts:
        - Page title
        - All text content
        - All links with their href attributes
        
        Args:
            content (str): Raw HTML content to extract data from
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'title': Page title or empty string if not found
                - 'text': All text content from the page
                - 'links': List of all href attributes from anchor tags
        """
        soup = BeautifulSoup(content, 'html.parser')
        return {
            'title': soup.title.string if soup.title else '',
            'text': soup.get_text(),
            'links': [a['href'] for a in soup.find_all('a', href=True)],
        }
