import asyncio
import logging
import os
import random
from functools import wraps
from typing import Callable, Optional, TypeVar

import dspy
import litellm
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_on_rate_limit(
    max_retries: int = 10,
    base_delay: float = 10.0,
    max_delay: float = 120.0,
):
    """
    Decorator that retries async functions on rate limit and transient errors with exponential backoff.
    Only sleeps when a retryable error is encountered.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds (will be multiplied by 2^attempt)
        max_delay: Maximum delay between retries in seconds
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except litellm.RateLimitError as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"Rate limit exceeded after {max_retries} retries: {e}"
                        )
                        raise
                    # Exponential backoff with jitter for rate limits
                    delay = min(base_delay * (2**attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.1)
                    sleep_time = delay + jitter
                    logger.warning(
                        f"Rate limit hit, sleeping {sleep_time:.1f}s before retry {attempt + 1}/{max_retries}"
                    )
                    await asyncio.sleep(sleep_time)
                except litellm.InternalServerError as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"Internal server error after {max_retries} retries: {e}"
                        )
                        raise
                    # Shorter delay for transient network errors
                    delay = min(2.0 * (2**attempt), 30.0)
                    jitter = random.uniform(0, delay * 0.1)
                    sleep_time = delay + jitter
                    logger.warning(
                        f"Internal server error (network issue), sleeping {sleep_time:.1f}s before retry {attempt + 1}/{max_retries}"
                    )
                    await asyncio.sleep(sleep_time)
            raise last_exception

        return wrapper

    return decorator


class LMConfig(BaseModel):
    """
    Configuration for the language model.
    """

    provider: str
    model_name: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None

    def get_model(self) -> dspy.LM:
        """
        Get a language model instance based on the configuration.

        Returns:
            dspy.LM: Configured language model instance
        """
        return LMFactory.get_model(
            self.provider,
            self.model_name,
            api_base=self.api_base,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
        )


class LMFactory:
    @staticmethod
    def get_model(
        provider: str,
        model_name: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> dspy.LM:
        """
        Get a language model instance.

        Args:
            provider: Provider of the language model ('openai', 'local')
            model_name: Name of the model
            api_base: Base URL of the API (required for VLLM models)
            **kwargs: Additional keyword arguments

        Returns:
            dspy.LM: Configured language model instance
        """
        if provider == "openai":
            return get_openai_model(model_name, max_tokens)
        elif provider == "local":
            if api_base is None:
                raise ValueError("api_base must be provided for local models")
            return get_local_model(model_name, api_base, api_key, max_tokens)
        elif provider == "together":
            return get_together_model(model_name, api_key)
        elif provider == "anthropic":
            return get_anthropic_model(model_name, max_tokens)
        elif provider == "vertex_ai":
            return get_vertex_model(model_name, max_tokens)
        else:
            raise ValueError(
                f"Unsupported provider or model: provider: {provider}, model: {model_name}"
            )


def get_openai_model(model_name: str, max_tokens: Optional[int]) -> dspy.LM:
    """
    Get an OpenAI language model instance.

    Args:
        model_name: Name of the model ('gpt-4o-mini', 'gpt-4o', 'o3-mini', 'o1-mini', 'o1-preview')

    Returns:
        dspy.LM: Configured language model instance
    """
    if max_tokens is None:
        max_tokens = 10000
        use_default_max_tokens = True
    else:
        use_default_max_tokens = False
    if model_name == "o4-mini":
        max_tokens = 20000
    model_configs = {
        "gpt-4o-mini": {"temperature": 1.0, "max_tokens": None},
        "gpt-4o": {"temperature": 1.0, "max_tokens": None},
        "o3-mini": {"temperature": 1.0, "max_tokens": max_tokens},
        "o1-mini": {"temperature": 1.0, "max_tokens": max_tokens},
        "o1": {"temperature": 1.0, "max_tokens": max_tokens},
        "o1-preview": {"temperature": 1.0, "max_tokens": max_tokens},
        "o4-mini": {"temperature": 1.0, "max_tokens": max_tokens},
        "gpt-4.1": {"temperature": 1.0, "max_tokens": None},
        "gpt-4.1-nano": {"temperature": 1.0, "max_tokens": None},
    }

    if model_name not in model_configs:
        raise ValueError(f"Unsupported model: {model_name}")

    config = model_configs[model_name]
    kwargs = {k: v for k, v in config.items() if v is not None}

    lm = dspy.LM(f"openai/{model_name}", cache=False, **kwargs)
    if use_default_max_tokens:
        if model_name == "o4-mini":
            lm.kwargs["max_completion_tokens"] = (
                10000  # ensure o4-mini uses the default max tokens
            )
    return lm


def get_local_model(
    model_name: str, api_base: str, api_key: Optional[str], max_tokens: Optional[int]
) -> dspy.LM:
    """
    Get a local model instance.

    Args:
        model_name: Name of the model
        api_base: Base URL of the API

    Returns:
        dspy.LM: Configured language model instance
    """
    if max_tokens is None:
        max_tokens = 10000
    return dspy.LM(
        f"hosted_vllm/{model_name}",
        api_base=api_base,
        api_key=api_key,
        temperature=1.0,
        max_tokens=max_tokens,
        cache=False,
    )


def get_together_model(model_name: str, api_key: Optional[str]) -> dspy.LM:
    """
    Get a Together AI language model instance.

    Args:
        model_name: Name of the model
        api_key: Together AI API key

    Returns:
        dspy.LM: Configured language model instance
    """
    model_configs = {
        "deepseek-ai/DeepSeek-R1": {"temperature": 1.0, "max_tokens": 5000},
    }

    if model_name not in model_configs:
        raise ValueError(f"Unsupported model: {model_name}")

    config = model_configs[model_name]
    kwargs = {k: v for k, v in config.items() if v is not None}

    return dspy.LM(
        f"together_ai/{model_name}",
        api_key=api_key or os.getenv("TOGETHER_API_KEY"),
        cache=False,
        **kwargs,
    )


def get_anthropic_model(model_name: str, max_tokens: Optional[int]) -> dspy.LM:
    """
    Get an Anthropic language model instance.
    Args:
        model_name: Name of the model
    Returns:
        dspy.LM: Configured language model instance
    """
    if max_tokens is None:
        max_tokens = 10000
    model_configs = {
        "claude-3-7-sonnet-20250219": {"temperature": 1.0, "max_tokens": max_tokens},
        "claude-sonnet-4-5-20250514": {"temperature": 1.0, "max_tokens": max_tokens},
        "claude-sonnet-4-5-20250929": {"temperature": 1.0, "max_tokens": max_tokens},
        "claude-haiku-4-5-20251001": {"temperature": 1.0, "max_tokens": max_tokens},
    }

    if model_name not in model_configs:
        raise ValueError(f"Unsupported model: {model_name}")

    config = model_configs[model_name]
    kwargs = {k: v for k, v in config.items() if v is not None}

    return dspy.LM(f"anthropic/{model_name}", cache=False, **kwargs)


def get_vertex_model(model_name: str, max_tokens: Optional[int]) -> dspy.LM:
    """
    Get a Vertex AI language model instance.

    Args:
        model_name: Name of the model

    Returns:
        dspy.LM: Configured language model instance
    """
    if max_tokens is None:
        max_tokens = 10000
    model_configs = {
        "gemini-2.5-flash-preview-04-17": {
            "temperature": 1.0,
            "max_tokens": max_tokens,
        },
        "gemini-2.5-pro-preview-05-06": {"temperature": 1.0, "max_tokens": max_tokens},
    }

    if model_name not in model_configs:
        raise ValueError(f"Unsupported model: {model_name}")

    config = model_configs[model_name]
    kwargs = {k: v for k, v in config.items() if v is not None}

    return dspy.LM(
        f"vertex_ai/{model_name}",
        cache=False,
        **kwargs,
    )
