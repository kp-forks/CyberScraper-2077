"""Mixins providing shared functionality for scrapers."""

from bs4 import BeautifulSoup


class HTMLExtractorMixin:
    """Mixin providing common HTML extraction methods."""

    @staticmethod
    def extract_html_data(content: str) -> dict:
        """
        Extract structured data from HTML content.

        Args:
            content: Raw HTML content to extract data from

        Returns:
            Dictionary containing:
                - 'title': Page title or empty string if not found
                - 'text': All text content from the page
                - 'links': List of all href attributes from anchor tags
        """
        soup = BeautifulSoup(content, 'lxml')
        return {
            'title': soup.title.string if soup.title else '',
            'text': soup.get_text(),
            'links': [a['href'] for a in soup.find_all('a', href=True)],
        }
