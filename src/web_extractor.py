from __future__ import annotations

import json
import pandas as pd
from io import StringIO, BytesIO
import base64
import re
from functools import lru_cache
from async_lru import alru_cache
import hashlib
import logging
import csv
import os

import tiktoken
from bs4 import BeautifulSoup, Comment
from urllib.parse import urlparse
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import Models
from .ollama_models import OllamaModel, OllamaModelManager
from .scrapers.playwright_scraper import PlaywrightScraper, ScraperConfig
from .scrapers.html_scraper import HTMLScraper
from .scrapers.json_scraper import JSONScraper
from .utils.proxy_manager import ProxyManager
from .utils.markdown_formatter import MarkdownFormatter
from .utils.error_handler import ErrorMessages, check_model_api_key
from .prompts import get_prompt_for_model
from .scrapers.tor.tor_scraper import TorScraper
from .scrapers.tor.tor_config import TorConfig
from .scrapers.tor.exceptions import TorException

logger = logging.getLogger(__name__)

# Module-level cached tiktoken encoding (singleton pattern)
_TIKTOKEN_ENCODING: tiktoken.Encoding | None = None


def _get_tiktoken_encoding() -> tiktoken.Encoding:
    """Get or create cached tiktoken encoding. Saves ~100-200ms per call."""
    global _TIKTOKEN_ENCODING
    if _TIKTOKEN_ENCODING is None:
        _TIKTOKEN_ENCODING = tiktoken.encoding_for_model("gpt-4o-mini")
    return _TIKTOKEN_ENCODING


# Precompiled regex patterns for JSON extraction
_JSON_BLOCK_PATTERN = re.compile(r'```json\s*([\s\S]*?)\s*```')
_CODE_BLOCK_PATTERN = re.compile(r'```\s*([\s\S]*?)\s*```')
# URL extraction pattern
_URL_PATTERN = re.compile(r'https?://[^\s/$.?#][^\s]*', re.IGNORECASE)


def extract_url(text: str) -> str | None:
    """Extract URL from anywhere in the text using regex."""
    match = _URL_PATTERN.search(text)
    return match.group(0) if match else None

# Tags to remove during preprocessing (single pass)
_REMOVE_TAGS = frozenset(['script', 'style', 'header', 'footer', 'nav', 'aside'])

class WebExtractor:
    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        model_kwargs: dict | None = None,
        proxy: str | None = None,
        scraper_config: ScraperConfig | None = None,
        tor_config: TorConfig | None = None
    ):
        model_kwargs = model_kwargs or {}

        # Check for required API keys before initializing
        api_key_error = check_model_api_key(model_name)
        if api_key_error:
            logger.warning(api_key_error)

        if isinstance(model_name, str) and model_name.startswith("ollama:"):
            self.model = OllamaModelManager.get_model(model_name[7:])
        elif isinstance(model_name, OllamaModel):
            self.model = model_name
        elif model_name.startswith("gemini-"):
            self.model = ChatGoogleGenerativeAI(model=model_name, **model_kwargs)
        else:
            self.model = Models.get_model(model_name, **model_kwargs)

        self.model_name = model_name
        self.scraper_config = scraper_config or ScraperConfig()
        self.playwright_scraper = PlaywrightScraper(config=self.scraper_config)
        self.html_scraper = HTMLScraper()
        self.json_scraper = JSONScraper()
        self.proxy_manager = ProxyManager(proxy)
        self.markdown_formatter = MarkdownFormatter()
        self.current_url: str | None = None
        self.current_content: str | None = None
        self.preprocessed_content: str | None = None
        self.conversation_history: list[str] = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=32000,
            chunk_overlap=200,
            length_function=self.num_tokens_from_string,
        )
        self.max_tokens = 128000 if model_name in ("gpt-4.1-mini", "gpt-4o-mini") else 16385
        self.query_cache: dict[tuple, str] = {}
        self.content_hash: str | None = None
        self.tor_config = tor_config or TorConfig()
        self.tor_scraper = TorScraper(self.tor_config)

    @staticmethod
    def num_tokens_from_string(string: str) -> int:
        encoding = _get_tiktoken_encoding()
        return len(encoding.encode(string))

    def _hash_content(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()

    def get_website_name(self, url: str) -> str:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain.split('.')[0].capitalize()

    async def _call_model(self, query: str) -> str:
        """Call the model to extract information from preprocessed content."""
        prompt_template = get_prompt_for_model(self.model_name)

        if isinstance(self.model, OllamaModel):
            full_prompt = prompt_template.format(
                webpage_content=self.preprocessed_content,
                query=query
            )
            return await self.model.generate(prompt=full_prompt)
        else:
            chain = prompt_template | self.model
            response = await chain.ainvoke({
                "webpage_content": self.preprocessed_content,
                "query": query
            })
            return response.content

    @staticmethod
    def _is_page_spec(value: str) -> bool:
        """Check if a string is a valid page specification (e.g., '1-5', '1,3,5', '2')."""
        if not value:
            return False
        # Valid page specs contain only digits, dashes, and commas
        return all(c.isdigit() or c in '-,' for c in value) and any(c.isdigit() for c in value)

    async def process_query(self, user_input: str, progress_callback=None) -> str:
        url = extract_url(user_input)
        if url:
            # Get text after the URL for parsing parameters
            url_match = _URL_PATTERN.search(user_input)
            text_after_url = user_input[url_match.end():].strip()

            parts = text_after_url.split(maxsplit=2)
            # Only treat as pages if it looks like a page specification (e.g., "1-5", "1,3,5")
            pages = parts[0] if len(parts) > 0 and self._is_page_spec(parts[0]) else None
            url_pattern = parts[1] if len(parts) > 1 and not parts[1].startswith('-') else None
            handle_captcha = '-captcha' in user_input.lower()

            website_name = self.get_website_name(url)

            if progress_callback:
                progress_callback(f"Fetching content from {website_name}...")

            response = await self._fetch_url(url, pages, url_pattern, handle_captcha, progress_callback)
        elif not self.current_content:
            response = "Please provide a URL first before asking for information."
        else:
            if progress_callback:
                progress_callback("Extracting information...")
            response = await self._extract_info(user_input)

        self.conversation_history.append(f"Human: {user_input}")
        self.conversation_history.append(f"AI: {response}")
        return response

    async def _fetch_url(self, url: str, pages: Optional[str] = None,
                        url_pattern: Optional[str] = None,
                        handle_captcha: bool = False,
                        progress_callback=None) -> str:
        self.current_url = url

        try:
            # Check if it's an onion URL
            if TorScraper.is_onion_url(url):
                if progress_callback:
                    progress_callback("Fetching content through Tor network...")

                content = await self.tor_scraper.fetch_content(url)
                self.current_content = content

            else:
                # Regular scraping without Tor
                if progress_callback:
                    progress_callback(f"Fetching content from {url}")

                # Don't use proxy for non-onion URLs
                contents = await self.playwright_scraper.fetch_content(
                    url,
                    proxy=None,  # Explicitly set proxy to None for regular URLs
                    pages=pages,
                    url_pattern=url_pattern,
                    handle_captcha=handle_captcha
                )

                # Check if scraping failed - only match if content starts with "Error:"
                # (not just contains it, as HTML pages often have "Error:" in scripts)
                if contents and any(str(c).strip().startswith("Error:") for c in contents):
                    return f"{ErrorMessages.SCRAPING_FAILED}\n\nDetails: {' '.join(contents)}"

                self.current_content = "\n".join(contents)

            if progress_callback:
                progress_callback("Preprocessing content...")

            self.preprocessed_content = self._preprocess_content(self.current_content)

            new_hash = self._hash_content(self.preprocessed_content)
            if self.content_hash != new_hash:
                self.content_hash = new_hash
                self.query_cache.clear()

            source_type = "Tor network" if TorScraper.is_onion_url(url) else "regular web"
            return f"I've fetched and preprocessed the content from {self.current_url} via {source_type}" + \
                (f" (pages: {pages})" if pages else "") + \
                ". What would you like to know about it?"

        except TorException as e:
            return str(e)
        except Exception as e:
            logger.error(f"Error fetching content: {str(e)}")
            return f"{ErrorMessages.SCRAPING_FAILED}\n\nDetails: {str(e)}"

    def _preprocess_content(self, content: str) -> str:
        # Use lxml parser for better performance
        soup = BeautifulSoup(content, 'lxml')

        # Single pass: remove unwanted tags and comments
        for element in soup.find_all(_REMOVE_TAGS):
            element.decompose()

        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Remove empty tags in one pass
        for tag in soup.find_all():
            if len(tag.get_text(strip=True)) == 0:
                tag.extract()

        text = soup.get_text()

        # Efficient text cleanup using generator expressions
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return '\n'.join(chunk for chunk in chunks if chunk)

    async def _extract_info(self, query: str) -> str:
        if not self.preprocessed_content:
            return "Please provide a URL first before asking for information."

        content_hash = self._hash_content(self.preprocessed_content)

        if self.content_hash != content_hash:
            self.content_hash = content_hash
            self.query_cache.clear()

        # Cache key includes model_name to prevent cross-model cache hits
        cache_key = (content_hash, query, self.model_name)

        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        content_tokens = self.num_tokens_from_string(self.preprocessed_content)

        if content_tokens <= self.max_tokens - 1000:
            extracted_data = await self._call_model(query)
        else:
            chunks = self.optimized_text_splitter(self.preprocessed_content)
            # Store original content, process chunks, restore
            original_content = self.preprocessed_content
            all_extracted_data = []
            for chunk in chunks:
                self.preprocessed_content = chunk
                chunk_data = await self._call_model(query)
                all_extracted_data.append(chunk_data)
            self.preprocessed_content = original_content
            extracted_data = self._merge_json_chunks(all_extracted_data)

        formatted_result = self._format_result(extracted_data, query)
        self.query_cache[cache_key] = formatted_result
        return formatted_result

    def _format_result(self, extracted_data: str, query: str) -> str | tuple[str, pd.DataFrame] | BytesIO:
        try:
            json_data = json.loads(extracted_data)
            
            if 'json' in query.lower():
                return self._format_as_json(json.dumps(json_data))
            elif 'csv' in query.lower():
                csv_string, df = self._format_as_csv(json.dumps(json_data))
                return f"```csv\n{csv_string}\n```", df
            elif 'excel' in query.lower():
                return self._format_as_excel(json.dumps(json_data))
            elif 'sql' in query.lower():
                return self._format_as_sql(json.dumps(json_data))
            elif 'html' in query.lower():
                return self._format_as_html(json.dumps(json_data))
            else:
                if isinstance(json_data, list) and all(isinstance(item, dict) for item in json_data):
                    csv_string, df = self._format_as_csv(json.dumps(json_data))
                    return f"```csv\n{csv_string}\n```", df
                else:
                    return self._format_as_json(json.dumps(json_data))
        
        except json.JSONDecodeError:
            return self._format_as_text(extracted_data)

    def optimized_text_splitter(self, text: str) -> List[str]:
        return self.text_splitter.split_text(text)

    def _merge_json_chunks(self, chunks: List[str]) -> str:
        merged_data = []
        for chunk in chunks:
            try:
                data = json.loads(chunk)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    merged_data.append(data)
            except json.JSONDecodeError:
                print(f"Error decoding JSON chunk: {chunk[:100]}...")
        return json.dumps(merged_data)

    @staticmethod
    def _extract_json_from_markdown(data: str) -> str:
        """Extract JSON content from markdown code blocks using precompiled patterns."""
        if match := _JSON_BLOCK_PATTERN.search(data):
            return match.group(1)
        if match := _CODE_BLOCK_PATTERN.search(data):
            return match.group(1)
        return data

    def _format_as_json(self, data: str) -> str:
        data = self._extract_json_from_markdown(data)
        try:
            parsed_data = json.loads(data)
            return f"```json\n{json.dumps(parsed_data, indent=2)}\n```"
        except json.JSONDecodeError:
            return f"Error: Invalid JSON data. Raw data: {data[:500]}..."

    def _format_as_csv(self, data: str) -> tuple[str, pd.DataFrame]:
        data = self._extract_json_from_markdown(data)
        try:
            parsed_data = json.loads(data)
            if not parsed_data:
                return "No data to convert to CSV.", pd.DataFrame()

            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=parsed_data[0].keys())
            writer.writeheader()
            writer.writerows(parsed_data)
            csv_string = output.getvalue()

            df = pd.DataFrame(parsed_data)

            return csv_string, df
        except json.JSONDecodeError:
            error_msg = f"Error: Invalid JSON data. Raw data: {data[:500]}..."
            return error_msg, pd.DataFrame()
        except Exception as e:
            error_msg = f"Error: Failed to convert data to CSV. {str(e)}"
            return error_msg, pd.DataFrame()

    def _format_as_excel(self, data: str) -> tuple[BytesIO, pd.DataFrame]:
        data = self._extract_json_from_markdown(data)
        try:
            parsed_data = json.loads(data)
            if not parsed_data:
                return BytesIO(b"No data to convert to Excel."), pd.DataFrame()

            df = pd.DataFrame(parsed_data)
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
            excel_buffer.seek(0)

            return excel_buffer, df
        except json.JSONDecodeError:
            error_msg = f"Error: Invalid JSON data. Raw data: {data[:500]}..."
            return BytesIO(error_msg.encode()), pd.DataFrame()
        except Exception as e:
            error_msg = f"Error: Failed to convert data to Excel. {str(e)}"
            return BytesIO(error_msg.encode()), pd.DataFrame()

    def _format_as_sql(self, data: str) -> str:
        data = self._extract_json_from_markdown(data)
        try:
            parsed_data = json.loads(data)
            if not parsed_data:
                return "No data to convert to SQL."

            fields = ", ".join([f"{k} TEXT" for k in parsed_data[0].keys()])
            sql_parts = [f"CREATE TABLE extracted_data ({fields});"]

            for row in parsed_data:
                escaped_values = [str(v).replace("'", "''") for v in row.values()]
                values = ", ".join([f"'{v}'" for v in escaped_values])
                sql_parts.append(f"INSERT INTO extracted_data VALUES ({values});")

            return f"```sql\n{chr(10).join(sql_parts)}\n```"
        except json.JSONDecodeError:
            return f"Error: Invalid JSON data. Raw data: {data[:500]}..."

    def _format_as_html(self, data: str) -> str:
        data = self._extract_json_from_markdown(data)
        try:
            parsed_data = json.loads(data)
            if not parsed_data:
                return "No data to convert to HTML."

            html_parts = ["<table>", "<tr>"]
            html_parts.extend([f"<th>{k}</th>" for k in parsed_data[0].keys()])
            html_parts.append("</tr>")

            for row in parsed_data:
                html_parts.append("<tr>")
                html_parts.extend([f"<td>{v}</td>" for v in row.values()])
                html_parts.append("</tr>")

            html_parts.append("</table>")

            return f"```html\n{''.join(html_parts)}\n```"
        except json.JSONDecodeError:
            return f"Error: Invalid JSON data. Raw data: {data[:500]}..."

    def _format_as_text(self, data: str) -> str:
        data = self._extract_json_from_markdown(data)
        try:
            parsed_data = json.loads(data)
            return "\n".join([", ".join([f"{k}: {v}" for k, v in item.items()]) for item in parsed_data])
        except json.JSONDecodeError:
            return data

    def format_to_markdown(self, text: str) -> str:
        return self.markdown_formatter.to_markdown(text)

    def format_from_markdown(self, markdown_text: str) -> str:
        return self.markdown_formatter.from_markdown(markdown_text)

    @staticmethod
    async def list_ollama_models() -> List[str]:
        return await OllamaModel.list_models()