
"""
JSON Scraper Module

This module provides functionality for extracting structured data from JSON content.
"""

import json
from .base_scraper import BaseScraper
from typing import Dict, Any


class JSONScraper(BaseScraper):
    """
    JSON Scraper Implementation
    
    This scraper extracts structured data from JSON content.
    Note that fetching JSON content is handled by PlaywrightScraper.
    """
    
    async def fetch_content(self, url: str, proxy: str = None) -> str:
        """
        Fetch JSON content from a given URL.
        
        Note: This method is not implemented in this class as JSON content
        fetching is handled by PlaywrightScraper.
        
        Args:
            url (str): The URL to fetch content from
            proxy (str, optional): Proxy server to use for the request
            
        Raises:
            NotImplementedError: Always raised as this method is not implemented here
        """
        raise NotImplementedError("JSON content is fetched by PlaywrightScraper")

    async def extract(self, content: str) -> Dict[str, Any]:
        """
        Extract structured data from JSON content.
        
        This method parses JSON content and converts it to a Python dictionary.
        If the content is not valid JSON, an error message is returned.
        
        Args:
            content (str): Raw JSON content to extract data from
            
        Returns:
            Dict[str, Any]: Parsed JSON data or error message if parsing fails
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON content"}
