
"""
Base Scraper Module

This module defines the abstract base class for all scrapers in the CyberScraper-2077 project.
All scraper implementations should inherit from BaseScraper and implement its abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseScraper(ABC):
    """
    Abstract base class for all scraper implementations.
    
    This class defines the common interface that all scrapers must implement.
    """
    
    @abstractmethod
    async def fetch_content(self, url: str, proxy: str = None) -> str:
        """
        Fetch content from a given URL.
        
        Args:
            url (str): The URL to fetch content from
            proxy (str, optional): Proxy server to use for the request
            
        Returns:
            str: The raw content fetched from the URL
        """
        pass

    @abstractmethod
    async def extract(self, content: str) -> Dict[str, Any]:
        """
        Extract structured data from raw content.
        
        Args:
            content (str): Raw content to extract data from
            
        Returns:
            Dict[str, Any]: Structured data extracted from the content
        """
        pass
