import aiohttp
import os
import json
import logging

from .utils.error_handler import ErrorMessages

logger = logging.getLogger(__name__)

# Session singleton for connection pooling
_session: aiohttp.ClientSession | None = None


async def _get_session() -> aiohttp.ClientSession:
    """Get or create a shared aiohttp session for connection pooling."""
    global _session
    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(total=120)
        _session = aiohttp.ClientSession(timeout=timeout)
    return _session


async def close_session() -> None:
    """Close the shared session. Call on application shutdown."""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None


class OllamaModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        try:
            session = await _get_session()
            async with session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": True
                }
            ) as response:
                response.raise_for_status()

                # Use list + join to avoid O(n²) string concatenation
                response_parts: list[str] = []
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'response' in data:
                                response_parts.append(data['response'])
                        except json.JSONDecodeError:
                            logger.warning(f"Error decoding JSON: {line}")

                return ''.join(response_parts)
        except aiohttp.ClientConnectorError:
            logger.error(ErrorMessages.OLLAMA_NOT_RUNNING)
            raise Exception(ErrorMessages.OLLAMA_NOT_RUNNING)
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                logger.error(f"Model {self.model_name} not found")
                raise Exception(ErrorMessages.OLLAMA_MODEL_NOT_FOUND)
            logger.error(f"HTTP error occurred: {str(e)}")
            raise Exception(ErrorMessages.OLLAMA_NOT_RUNNING)
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            raise Exception(ErrorMessages.OLLAMA_NOT_RUNNING)

    @staticmethod
    async def list_models() -> list[str]:
        base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        try:
            session = await _get_session()
            async with session.get(f"{base_url}/api/tags") as response:
                response.raise_for_status()
                models = await response.json()
                return [model['name'] for model in models.get('models', [])]
        except aiohttp.ClientConnectorError:
            logger.warning(ErrorMessages.OLLAMA_NOT_RUNNING)
            return []
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {str(e)}")
            return []


class OllamaModelManager:
    @staticmethod
    def get_model(model_name: str) -> OllamaModel:
        return OllamaModel(model_name)