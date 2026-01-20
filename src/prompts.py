"""
Prompt templates for web scraping extraction.

Consolidated prompts with shared base instructions to reduce token usage (~40% reduction).
"""

from langchain_core.prompts import PromptTemplate

# Shared base instructions (used by all models)
_BASE_INSTRUCTIONS = """You are an AI assistant for web scraping tasks.
Extract information from the webpage content based on the user's query.
Always return a JSON array of objects. Each object represents one item/row.

Format (no additional text):
[
  {{"field1": "value1", "field2": "value2"}},
  {{"field1": "value3", "field2": "value4"}}
]

Rules:
- For data questions: provide bullet-point summary and use cases
- Include all requested fields; use "N/A" if not found
- Never invent data not present in the content
- Limit entries if a specific count is requested
- Use relevant field names based on content and query

Webpage content:
{webpage_content}

Query: {query}
"""

# Create unified prompt template
_UNIFIED_PROMPT = PromptTemplate(
    input_variables=["webpage_content", "query"],
    template=_BASE_INSTRUCTIONS
)


def get_prompt_for_model(model_name: str) -> PromptTemplate:
    """
    Get the appropriate prompt template for a given model.

    All models now use the same consolidated prompt for consistency
    and reduced token usage.

    Args:
        model_name: The name of the model (e.g., "gpt-4o-mini", "gemini-pro", "ollama:llama2")

    Returns:
        PromptTemplate configured for the model

    Raises:
        ValueError: If the model is not supported
    """
    match model_name:
        case name if name.startswith(("gpt-", "text-")):
            return _UNIFIED_PROMPT
        case name if name.startswith("gemini-"):
            return _UNIFIED_PROMPT
        case name if name.startswith("ollama:"):
            return _UNIFIED_PROMPT
        case _:
            raise ValueError(f"Unsupported model: {model_name}")
