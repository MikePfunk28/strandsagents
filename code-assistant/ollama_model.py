"""Ollama model integration for StrandsAgents coding assistant."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class OllamaModel:
    """Ollama model wrapper for StrandsAgents compatible interface."""

    def __init__(
        self,
        model_name: str = "llama3.2",
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.endpoint = f"{self.host}/api/generate"
        self.chat_endpoint = f"{self.host}/api/chat"

        logger.info("OllamaModel initialized: %s at %s", model_name, host)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[List[int]] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """Generate text using Ollama generate endpoint."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }

        if system:
            payload["system"] = system

        if context:
            payload["context"] = context

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()

            if stream:
                # Handle streaming response
                full_response = ""
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            full_response += data["response"]
                        if data.get("done", False):
                            break
                return full_response
            else:
                # Handle non-streaming response
                data = response.json()
                return data.get("response", "")

        except requests.exceptions.RequestException as e:
            logger.error("Ollama request failed: %s", e)
            raise RuntimeError(f"Ollama request failed: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> str:
        """Generate text using Ollama chat endpoint."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }

        try:
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()

            if stream:
                # Handle streaming response
                full_response = ""
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            full_response += data["message"]["content"]
                        if data.get("done", False):
                            break
                return full_response
            else:
                # Handle non-streaming response
                data = response.json()
                message = data.get("message", {})
                return message.get("content", "")

        except requests.exceptions.RequestException as e:
            logger.error("Ollama chat request failed: %s", e)
            raise RuntimeError(f"Ollama chat request failed: {e}")

    def embed(self, text: str) -> List[float]:
        """Generate embeddings using Ollama embeddings endpoint."""
        embed_endpoint = f"{self.host}/api/embeddings"
        payload = {
            "model": os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma"),
            "prompt": text
        }

        try:
            response = requests.post(
                embed_endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])

        except requests.exceptions.RequestException as e:
            logger.error("Ollama embedding request failed: %s", e)
            raise RuntimeError(f"Ollama embedding request failed: {e}")

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models on Ollama server."""
        list_endpoint = f"{self.host}/api/tags"

        try:
            response = requests.get(list_endpoint, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])

        except requests.exceptions.RequestException as e:
            logger.error("Failed to list Ollama models: %s", e)
            return []

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        pull_endpoint = f"{self.host}/api/pull"
        payload = {"name": model_name}

        try:
            response = requests.post(
                pull_endpoint,
                json=payload,
                timeout=300  # Longer timeout for model downloads
            )
            response.raise_for_status()
            return True

        except requests.exceptions.RequestException as e:
            logger.error("Failed to pull model %s: %s", model_name, e)
            return False

    def is_model_available(self, model_name: Optional[str] = None) -> bool:
        """Check if a model is available on the Ollama server."""
        model_to_check = model_name or self.model_name
        models = self.list_models()
        return any(model["name"].startswith(model_to_check) for model in models)

    def health_check(self) -> bool:
        """Check if Ollama server is healthy."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    # StrandsAgents compatibility methods
    def invoke(self, prompt: str, **kwargs) -> str:
        """StrandsAgents compatible invoke method."""
        system = kwargs.get("system")
        return self.generate(prompt, system=system, **kwargs)

    def stream_invoke(self, prompt: str, **kwargs):
        """StrandsAgents compatible streaming invoke method."""
        system = kwargs.get("system")
        # For streaming, we'll use the generate method with stream=True
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }

        if system:
            payload["system"] = system

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break

        except requests.exceptions.RequestException as e:
            logger.error("Ollama streaming request failed: %s", e)
            raise RuntimeError(f"Ollama streaming request failed: {e}")

    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model_name,
            "host": self.host,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "available": self.is_model_available(),
            "server_healthy": self.health_check()
        }


def create_ollama_model(
    model_name: str = "llama3.2",
    host: Optional[str] = None,
    **kwargs
) -> OllamaModel:
    """Factory function to create OllamaModel instance."""
    host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return OllamaModel(model_name=model_name, host=host, **kwargs)