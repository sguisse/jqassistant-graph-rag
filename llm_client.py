#!/usr/bin/env python3
"""
This module provides a client for interacting with various LLM APIs.
"""

import os
import logging
import shlex
import subprocess
import requests  # NOTE: This script requires the 'requests' library to be installed.
import threading


# Thread-safe global progress tracker for LLM prompts
class _GlobalProgress:
    def __init__(self):
        self.lock = threading.Lock()
        self.total: int | None = None
        self.current: int = 0

    def set_total(self, total: int) -> None:
        with self.lock:
            self.total = int(total) if total is not None else None
            self.current = 0

    def increment(self) -> int:
        with self.lock:
            self.current += 1
            return self.current

    def reset(self) -> None:
        with self.lock:
            self.total = None
            self.current = 0


# module-global tracker
_GLOBAL_LLM_PROGRESS = _GlobalProgress()


def set_global_llm_total(total: int) -> None:
    """Set the global total number of LLM prompts for this run.

    Call this at the start of the run (if you can compute the total) so
    generate_summary can display global progress as Prompt X/Y.
    """
    _GLOBAL_LLM_PROGRESS.set_total(total)


def reset_global_llm_progress() -> None:
    """Reset the global progress counter and clear total."""
    _GLOBAL_LLM_PROGRESS.reset()


# ---------------------------------------------------------------------------
# Per-pass (per-label) progress tracker
# Tracks how many items have been processed within the current summarizer pass
# (e.g. "DirectorySummarizer: 42/300"). One _GlobalProgress instance per label.
# ---------------------------------------------------------------------------
_pass_progress: dict[str, _GlobalProgress] = {}
_pass_lock = threading.Lock()

# Lock that serializes all console output from generate_summary so that
# concurrent threads (ThreadPoolExecutor inside process_batch) never
# interleave their header / prompt / response lines. It is also used to
# keep the visible order of headers aligned with the order in which counter
# slots are reserved.
_print_lock = threading.Lock()


def set_pass_total(label: str, total: int) -> None:
    """Register the total number of items for the named summarizer pass.

    Called by BaseSummarizer.process_batch() before launching the thread pool.
    """
    with _pass_lock:
        if label not in _pass_progress:
            _pass_progress[label] = _GlobalProgress()
    _pass_progress[label].set_total(total)


def add_to_pass_total(label: str, n: int) -> None:
    """Add *n* items to the running total for *label* without resetting the
    current counter.  Used by multi-batch passes (e.g. DirectorySummarizer
    that iterates depth levels) so the counter accumulates across batches.
    """
    with _pass_lock:
        if label not in _pass_progress:
            _pass_progress[label] = _GlobalProgress()
        tracker = _pass_progress[label]
    with tracker.lock:
        if tracker.total is None:
            tracker.total = n
        else:
            tracker.total += n


def reset_pass_progress(label: str) -> None:
    """Reset the item counter for the named summarizer pass."""
    with _pass_lock:
        tracker = _pass_progress.get(label)
    if tracker:
        tracker.reset()


def reset_all_pass_progress() -> None:
    """Reset ALL per-label pass counters.  Call once at the start of a full run."""
    with _pass_lock:
        for tracker in _pass_progress.values():
            tracker.reset()


def _increment_pass(label: str) -> tuple[int, int | None]:
    """Increment the per-label pass counter and return (current, total)."""
    with _pass_lock:
        tracker = _pass_progress.get(label)
    if tracker:
        idx = tracker.increment()
        return idx, tracker.total
    return 0, None


# ---------------------------------------------------------------------------
# Per-pass LLM call quota (read from env vars at first use)
#
# Env var naming: CamelCase class name → UPPER_SNAKE + _MAX_LLM_CALL
#   MethodAnalyzer       → METHOD_ANALYZER_MAX_LLM_CALL
#   MethodSummarizer     → METHOD_SUMMARIZER_MAX_LLM_CALL
#   TypeSummarizer       → TYPE_SUMMARIZER_MAX_LLM_CALL
#   SourceFileSummarizer → SOURCE_FILE_SUMMARIZER_MAX_LLM_CALL
#   DirectorySummarizer  → DIRECTORY_SUMMARIZER_MAX_LLM_CALL
#   PackageSummarizer    → PACKAGE_SUMMARIZER_MAX_LLM_CALL
#   ProjectSummarizer    → PROJECT_SUMMARIZER_MAX_LLM_CALL
#
# A global fallback can also be set: MAX_LLM_CALL (applies when no label-
# specific var is defined). Set to 0 or leave unset to disable the quota.
# ---------------------------------------------------------------------------
_label_call_counts: dict[str, int] = {}
_label_call_lock = threading.Lock()

# Lock that serializes quota acceptance + progress slot reservation so that
# counters stay coherent even when generate_summary() is called concurrently.
_reservation_lock = threading.Lock()


def _camel_to_upper_snake(name: str) -> str:
    """Convert CamelCase to UPPER_SNAKE_CASE.

    Examples:
        MethodAnalyzer       → METHOD_ANALYZER
        SourceFileSummarizer → SOURCE_FILE_SUMMARIZER
    """
    import re

    # Insert underscore before each uppercase letter that follows a lowercase
    # letter or another uppercase letter followed by lowercase.
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s1)
    return s2.upper()


def _get_label_max(label: str) -> int | None:
    """Return the max LLM calls for *label*, or None if unlimited.

    Reads (in priority order):
      1. UPPER_SNAKE(label)_MAX_LLM_CALL  (label-specific)
      2. MAX_LLM_CALL                     (global fallback)
    Returns None when neither is set or when value is 0.
    """
    snake = _camel_to_upper_snake(label)
    raw = os.environ.get(f"{snake}_MAX_LLM_CALL") or os.environ.get("MAX_LLM_CALL")
    if raw:
        try:
            val = int(raw)
            return val if val > 0 else None
        except ValueError:
            pass
    return None


def _reserve_progress_slot(
    label: str, index: int, total: int
) -> tuple[str, int, int | None] | None:
    """Reserve one real LLM call slot and return display progress.

    This function is the single source of truth for accepting a call:
    - quota is checked first,
    - only accepted calls increment counters,
    - pass/global progress are reserved atomically.

    Returns:
        (local_progress_string, global_index, global_total) if accepted,
        or None when the call must be skipped because the quota is reached.
    """
    with _reservation_lock:
        if label:
            max_calls = _get_label_max(label)
            with _label_call_lock:
                current_calls = _label_call_counts.get(label, 0)
                if max_calls is not None and current_calls >= max_calls:
                    return None
                _label_call_counts[label] = current_calls + 1

            pass_idx, pass_total = _increment_pass(label)
            local_str = f"{pass_idx}/{pass_total}" if pass_total else f"{index}/{total}"
        else:
            local_str = f"{index}/{total}"

        global_index = _GLOBAL_LLM_PROGRESS.increment()
        global_total = _GLOBAL_LLM_PROGRESS.total
        return local_str, global_index, global_total


def reset_label_call_counts() -> None:
    """Reset all per-label LLM call counters (useful between runs)."""
    with _label_call_lock:
        _label_call_counts.clear()


logger = logging.getLogger(__name__)

# --- Summarization Clients ---


class LlmClient:
    """
    Base class for LLM clients.

    Implements a Template Method for generate_summary:
      1. Prints the prompt for console visibility.
      2. Delegates to _call_llm (overridden by each concrete subclass).
      3. Prints the response and a separator.
    Subclasses must implement _call_llm and must NOT print the prompt or
    the response themselves — that is handled here, once, for all clients.
    """

    is_local: bool = False

    def generate_summary(
        self, prompt: str, index: int = 1, total: int = 1, label: str = ""
    ) -> str:
        """
        Template Method: prints prompt with an index/total indicator,
        delegates to _call_llm and prints the response. Subclasses should
        implement _call_llm(prompt, index, total) and MUST NOT print the
        prompt/response themselves.

        Args:
            prompt: The LLM prompt text.
            index:  Local chunk index (1-based) within the current node's
                    iterative processing (e.g. chunk 2 of 3).
            total:  Total number of chunks for the current node.
            label:  Pass identifier shown in the console header, e.g.
                    "Analyzer", "MethodSummarizer", "TypeSummarizer", …
        """
        with _print_lock:
            reserved = _reserve_progress_slot(label, index, total)
            if reserved is None:
                if label:
                    env_var = f"{_camel_to_upper_snake(label)}_MAX_LLM_CALL"
                    max_calls = _get_label_max(label)
                    logger.debug(
                        "Skipping LLM call for %s: quota reached. Increase %s (current max=%s).",
                        label,
                        env_var,
                        max_calls,
                    )
                return ""

            local_str, global_index, global_total = reserved

            # Build the prefix: "💬 {label} - " when a label is provided, else "💬 ".
            prefix = f"\U0001f4ac {label} - " if label else "\U0001f4ac "

            # If a global total was set, show (local item/total) / (global index/total).
            # Otherwise show only the local progress.
            if global_total and global_total > 0:
                header = (
                    f"{prefix}Prompt ({local_str}) / ({global_index}/{global_total}) :"
                )
            else:
                header = f"{prefix}Prompt ({local_str}) :"

            print()
            print(header)
            print(prompt)
            print()

        result = self._call_llm(prompt, index, total)

        if result:
            with _print_lock:
                print()
                print(f"\u2728 LLM response received ({len(result)} chars): ")
                print(result)
                print("----------------------------")

        return result

    def _call_llm(self, prompt: str, index: int = 1, total: int = 1) -> str:
        """
        Override in subclasses to perform the actual LLM call. Receives the
        loop index and total so implementations can adapt logging if desired.
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

    def _call_llm(self, prompt: str, index: int = 1, total: int = 1) -> str:
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

    def _call_llm(self, prompt: str, index: int = 1, total: int = 1) -> str:
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

    def _call_llm(self, prompt: str, index: int = 1, total: int = 1) -> str:
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

    def _call_llm(self, prompt: str, index: int = 1, total: int = 1) -> str:

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
    A fake client for debugging that returns a context-aware stub summary.
    It parses the prompt to extract the node type and name so the returned
    message is more useful than a generic placeholder.
    """

    # is_local: bool = True

    # Ordered list of (regex, part_label, group_index_for_name)
    _PATTERNS: list[tuple] = []

    @staticmethod
    def _extract_part(prompt: str) -> tuple[str, str]:
        """Return (part, partName) extracted from *prompt*.

        Falls back to ("part", "") when no pattern matches.
        """
        import re

        patterns = [
            # MethodSummarizer single-shot
            (r"A method named '([^']+)'", "Method"),
            # TypeSummarizer single-shot: "A <label> named '<name>' is defined"
            (r"A ([\w]+) named '([^']+)' is defined", "Type"),
            # TypeSummarizer iterative
            (r"the ([\w]+) '([^']+)' is currently", "Type"),
            # MethodAnalyzer: code block, extract first identifier after function keyword
            (r"Summarize the purpose of this method", "MethodAnalysis"),
            # Hierarchical: directory / source file / package / project
            (r"for the directory named '([^']+)'", "Directory"),
            (r"for the source file named '([^']+)'", "SourceFile"),
            (r"for the package named '([^']+)'", "Package"),
            (r"for the project named '([^']+)'", "Project"),
            # Iterative hierarchical
            (
                r"the (Directory|SourceFile|Package|Project) '([^']+)' is currently",
                "Hierarchical",
            ),
            # Project summary prompt
            (r"for the project named '([^']+)'", "Project"),
        ]

        for pattern, kind in patterns:
            m = re.search(pattern, prompt)
            if m:
                if kind == "Type" and m.lastindex == 2:
                    type_label, type_name = m.group(1), m.group(2)
                    return f"{type_label.capitalize()}", type_name
                elif kind == "Hierarchical" and m.lastindex == 2:
                    return m.group(1), m.group(2)
                elif kind == "MethodAnalysis":
                    # Try to extract the method signature from the code block
                    sig = re.search(r"```[^\n]*\n([^\n]{0,120})", prompt)
                    name = sig.group(1).strip() if sig else "(code chunk)"
                    return "MethodAnalysis", name
                elif m.lastindex and m.lastindex >= 1:
                    return kind, m.group(1)

        return "part", ""

    def _call_llm(self, prompt: str, index: int = 1, total: int = 1) -> str:
        part, name = self._extract_part(prompt)
        if name:
            return f"This {part} '{name}' implements important functionalities."
        return f"This {part} implements important functionalities."


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
        if not os.path.isdir(model_name):
            bundled_model_dir = os.path.join(
                os.path.dirname(__file__), "models", model_name
            )
            if os.path.isdir(bundled_model_dir):
                model_name = bundled_model_dir
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
