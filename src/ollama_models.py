import aiohttp
import os
import json
import logging

from .utils.error_handler import ErrorMessages

logger = logging.getLogger(__name__)

# A short timeout for control-plane calls (model listing) and a generous one for generation,
# which can stream for minutes on large local models.
_LIST_TIMEOUT = aiohttp.ClientTimeout(total=10)
_GENERATE_TIMEOUT = aiohttp.ClientTimeout(total=600, sock_read=120)


class OllamaModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        # Streamlit calls asyncio.run() per request, so a singleton session would
        # be bound to a closed event loop on the second call. Open a fresh session
        # per call — there is no pool to reuse across requests anyway.
        try:
            async with aiohttp.ClientSession(timeout=_GENERATE_TIMEOUT) as session:
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
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Ollama unreachable at {self.base_url}: {e}")
            raise Exception(ErrorMessages.OLLAMA_NOT_RUNNING)
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                logger.error(f"Model {self.model_name} not found at {self.base_url}")
                raise Exception(ErrorMessages.OLLAMA_MODEL_NOT_FOUND)
            logger.error(f"Ollama HTTP {e.status}: {e.message}")
            raise Exception(f"Ollama HTTP {e.status}: {e.message}")
        except Exception as e:
            # Re-raise the actual error instead of masking it as "Ollama not running"
            logger.exception("Ollama generate failed")
            raise Exception(f"Ollama generate failed: {type(e).__name__}: {e}")

    @staticmethod
    async def list_models() -> list[str]:
        base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        try:
            async with aiohttp.ClientSession(timeout=_LIST_TIMEOUT) as session:
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