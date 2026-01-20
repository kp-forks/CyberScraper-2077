"""Markdown formatting utilities."""

import re
import markdown

# Precompiled patterns for markdown stripping (only match markdown syntax)
_HEADER_PATTERN = re.compile(r'^#{1,6}\s+', re.MULTILINE)
_BOLD_ASTERISK = re.compile(r'\*\*(.+?)\*\*')
_BOLD_UNDERSCORE = re.compile(r'__(.+?)__')
_ITALIC_ASTERISK = re.compile(r'\*(.+?)\*')
_ITALIC_UNDERSCORE = re.compile(r'_(.+?)_')
_STRIKETHROUGH = re.compile(r'~~(.+?)~~')
_INLINE_CODE = re.compile(r'`([^`]+)`')
_LINK_PATTERN = re.compile(r'\[([^\]]+)\]\([^)]+\)')


class MarkdownFormatter:
    """Utility class for markdown conversion."""

    @staticmethod
    def to_markdown(text: str) -> str:
        """Convert plain text to HTML using markdown."""
        return markdown.markdown(text)

    @staticmethod
    def from_markdown(markdown_text: str) -> str:
        """
        Strip markdown formatting from text.

        Uses regex patterns that only match markdown syntax,
        preserving text like 'C#' or 'snake_case'.
        """
        text = markdown_text

        # Remove headers (only at start of line with space after #)
        text = _HEADER_PATTERN.sub('', text)

        # Remove bold/italic (must have matching pairs)
        text = _BOLD_ASTERISK.sub(r'\1', text)
        text = _BOLD_UNDERSCORE.sub(r'\1', text)
        text = _ITALIC_ASTERISK.sub(r'\1', text)
        text = _ITALIC_UNDERSCORE.sub(r'\1', text)

        # Remove strikethrough
        text = _STRIKETHROUGH.sub(r'\1', text)

        # Remove inline code backticks
        text = _INLINE_CODE.sub(r'\1', text)

        # Extract link text
        text = _LINK_PATTERN.sub(r'\1', text)

        return text

