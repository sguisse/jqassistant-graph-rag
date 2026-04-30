#!/usr/bin/env python3
"""
This module provides a client for interacting with various LLM APIs.
"""

import os
import logging
import shlex
import subprocess
import requests  # NOTE: This script requires the 'requests' library to be installed.

logger = logging.getLogger(__name__)

# --- Summarization Clients ---


class LlmClient:
    """
    Base class for LLM clients.
    """

    is_local: bool = False

    def generate_summary(self, prompt: str) -> str:
        """
        Generates a summary for a given prompt.
        """
        raise NotImplementedError


class OpenAiClient(LlmClient):
    """
    Client for OpenAI's API.
    """

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

    def generate_summary(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=120
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            logger.error(f"OpenAI API request failed: {e}")
            return ""


class DeepSeekClient(LlmClient):
    """
    Client for DeepSeek's API.
    """

    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set.")
        self.api_url = "https://api.deepseek.com/chat/completions"
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-coder")

    def generate_summary(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=120
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            logger.error(f"DeepSeek API request failed: {e}")
            return ""


class OllamaClient(LlmClient):
    """
    Client for a local Ollama instance.
    """

    is_local: bool = True

    def __init__(self):
        # self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://xf-gpu.local:11434")
        if not self.base_url:
            raise ValueError("OLLAMA_BASE_URL environment variable not set.")
        self.api_url = f"{self.base_url.rstrip('/')}/api/generate"
        # TODO: the deepseek-r1:8b model generates response with tags like <think>...</think> that should be removed
        # self.model = os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b")
        self.model = os.environ.get("OLLAMA_MODEL", "deepseek-llm:7b")

    def generate_summary(self, prompt: str) -> str:
        return self.generate_summary_chat(prompt)

    def generate_summary_chat(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        response.raise_for_status()
        return response.json()["message"]["content"]

    def generate_summary_reasoning(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            response = requests.post(self.api_url, json=payload, timeout=300)
            response.raise_for_status()
            return response.json()["response"]
        except requests.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            return ""


class CliLlmClient(LlmClient):
    """
    Client that calls a local LLM CLI tool (e.g., gemini, copilot) by spawning a subprocess.
    The prompt is passed via the -p flag: `{cmd} [{extra_params}] -p "{prompt}"`.

    Configuration via environment variables:
      - LLM_CLI_CMD    : the CLI binary name or path  (default: 'gemini')
      - LLM_CLI_PARAMS : optional extra flags          (e.g. '--model gpt-5-mini --effort high')
      - LLM_CLI_TIMEOUT: subprocess timeout in seconds (default: 300)

    Examples:
      LLM_CLI_CMD=gemini
      LLM_CLI_CMD=copilot  LLM_CLI_PARAMS="--model gpt-5-mini --effort high"
    """

    is_local: bool = True

    def __init__(self):
        self.cli_cmd = os.environ.get("LLM_CLI_CMD", "gemini")
        self.cli_params = os.environ.get("LLM_CLI_PARAMS", "")
        self.timeout = int(os.environ.get("LLM_CLI_TIMEOUT", "300"))
        if not self.cli_cmd:
            raise ValueError("LLM_CLI_CMD environment variable not set.")
        params_display = f" {self.cli_params}" if self.cli_params else ""
        logger.info(
            f"CliLlmClient initialized: '{self.cli_cmd}{params_display} -p \"...\"'"
        )
        print(f"🤖 CliLlmClient ready: '{self.cli_cmd}{params_display} -p \"...\"'")

    def generate_summary(self, prompt: str) -> str:
        cmd_parts = [self.cli_cmd]
        if self.cli_params:
            cmd_parts.extend(shlex.split(self.cli_params))
        cmd_parts.extend(["-p", prompt])

        params_display = f" {self.cli_params}" if self.cli_params else ""
        cmd_display = f'{self.cli_cmd}{params_display} -p "..."'
        print(f"📡 Calling CLI: {cmd_display}")
        logger.info(f"Calling CLI: {cmd_display}")

        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode != 0:
                err = result.stderr.strip()
                print(f"⚠️  CLI exited with code {result.returncode}: {err}")
                logger.error(f"CLI error (exit {result.returncode}): {err}")
                return ""
            output = result.stdout.strip()
            print(f"✅ CLI response received ({len(output)} chars)")
            logger.info(f"CLI response received: {len(output)} chars")
            return output
        except subprocess.TimeoutExpired:
            print(f"⏰ CLI timed out after {self.timeout}s")
            logger.error(f"CLI command timed out after {self.timeout}s")
            return ""
        except FileNotFoundError:
            print(f"❌ CLI command not found: '{self.cli_cmd}'")
            logger.error(f"CLI command not found: '{self.cli_cmd}'")
            return ""
        except Exception as e:
            print(f"❌ CLI call failed: {e}")
            logger.error(f"CLI call failed: {e}")
            return ""


class FakeLlmClient(LlmClient):
    """
    A fake client for debugging that returns a static summary. Acts as remote API service
    """

    # is_local: bool = True

    def generate_summary(self, prompt: str) -> str:
        """
        Returns a hardcoded summary for any prompt.
        """
        return "This part implements important functionalities."


def get_llm_client(api_name: str) -> LlmClient:
    """
    Factory function to get an LLM client.
    """
    api_name = api_name.lower()
    if api_name == "openai":
        return OpenAiClient()
    elif api_name == "deepseek":
        return DeepSeekClient()
    elif api_name == "ollama":
        return OllamaClient()
    elif api_name == "cli":
        return CliLlmClient()
    elif api_name == "fake":
        return FakeLlmClient()
    else:
        raise ValueError(
            f"Unknown API: {api_name}. Supported APIs are: openai, deepseek, ollama, cli, fake."
        )


# --- Embedding Clients ---
# NOTE: The SentenceTransformerClient requires 'sentence-transformers' and 'torch'
# to be installed. Please run: pip install sentence-transformers


class EmbeddingClient:
    """
    Base class for embedding clients.
    """

    is_local: bool = False

    def generate_embeddings(
        self, texts: list[str], show_progress_bar: bool = True
    ) -> list[list[float]]:
        """
        Generates embedding vectors for a given list of texts.
        """
        raise NotImplementedError


class SentenceTransformerClient(EmbeddingClient):
    """
    Client that uses a local SentenceTransformer model.
    """

    is_local: bool = True

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "The 'sentence-transformers' package is required for local embeddings. Please run 'pip install sentence-transformers' to install it."
            )

        model_name = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        logger.info(f"Loading local SentenceTransformer model: {model_name}")
        # If the model name resolves to a local directory, pass local_files_only=True
        # so that SentenceTransformer (and the underlying transformers/huggingface_hub
        # stack) never attempts any network requests. This is more reliable than env
        # vars because TRANSFORMERS_OFFLINE is read at import time in some versions.
        is_local_dir = os.path.isdir(model_name)
        if is_local_dir:
            logger.info(
                "Local model directory detected – loading with local_files_only=True (no network calls)."
            )
        try:
            self.model = SentenceTransformer(model_name, local_files_only=is_local_dir)
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            # Detect SSL / certificate verification failures which commonly occur on
            # macOS or misconfigured environments when attempting to download from
            # Hugging Face. Provide actionable remediation steps instead of a raw
            # traceback so users know how to fix it quickly.
            import ssl

            err_str = str(e)
            is_cert_err = False
            try:
                is_cert_err = isinstance(e, ssl.SSLError)
            except Exception:
                is_cert_err = False
            if (
                "CERTIFICATE_VERIFY_FAILED" in err_str
                or "certificate verify failed" in err_str
                or is_cert_err
            ):
                msg = (
                    "❌ Failed to download the SentenceTransformer model due to an SSL certificate verification error.\n"
                    "Common fixes:\n"
                    "  1) Install/upgrade 'certifi' and point OpenSSL to it:\n"
                    "       python3 -m pip install --upgrade certifi\n"
                    "       export SSL_CERT_FILE=$(python3 -m certifi)\n"
                    "  2) On macOS with the system Python, run the 'Install Certificates.command' that ships with Python (one-time).\n"
                    "  3) Use a locally cached model instead of downloading by setting the env var:\n"
                    "       export SENTENCE_TRANSFORMER_MODEL=/path/to/local/model\n"
                    "After applying a fix, re-run the command inside the project's virtualenv:\n"
                    "  .venv/bin/python main.py [args]\n"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
            else:
                logger.critical(
                    f"❌ Failed to load SentenceTransformer model: {e}", exc_info=True
                )
                raise

    def generate_embeddings(
        self, texts: list[str], show_progress_bar: bool = True
    ) -> list[list[float]]:
        """
        Generates embedding vectors for a given list of texts.

        Args:
            texts: List of text strings to embed
            show_progress_bar: Whether to show a progress bar during encoding

        Returns:
            List of embedding vectors as lists of floats
        """
        # The encode method can show its own progress bar, which is useful for large batches.
        embeddings = self.model.encode(texts, show_progress_bar=show_progress_bar)
        # Convert numpy arrays to standard lists for JSON/Neo4j compatibility
        return [emb.tolist() for emb in embeddings]


def get_embedding_client(api_name: str) -> EmbeddingClient:
    """
    Factory function to get an embedding client.
    """
    # The api_name can be used in the future to select different embedding models/APIs
    # For now, we default to the local sentence-transformer for all cases.
    logger.info("Initializing local SentenceTransformer client for embeddings.")
    return SentenceTransformerClient()
