"""
LLM Helper Utilities for Modal Qwen2.5-14B Integration

This module provides utilities to interact with the self-hosted Qwen2.5-14B model
deployed on Modal. It handles HTTP communication, retry logic, and error handling.
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any
import httpx
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from src.config.settings import MAX_RETRIES

# Configure logging
logger = logging.getLogger(__name__)

# Modal LLM Endpoint
QWEN_ENDPOINT = "https://alaapocket3--qwen2-5-14b-instruct-qwenendpoint-generate.modal.run"


# Request configuration
DEFAULT_TIMEOUT = 120.0  # seconds (increased for complex queries)
RETRY_BACKOFF = 2.0  # exponential backoff multiplier


class LLMError(Exception):
    """Custom exception for LLM-related errors"""
    pass


@traceable(name="call_llm", run_type="llm")
async def call_llm(
    prompt: str,
    system_message: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: float = DEFAULT_TIMEOUT,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Call the Modal Qwen2.5-14B LLM endpoint with retry logic.
    
    Args:
        prompt: The user prompt/question
        system_message: Optional system prompt that defines the LLM's behavior
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens to generate (default: 512 for faster responses)
        timeout: Request timeout in seconds
        metadata: Optional metadata for tracing/logging
        
    Returns:
        str: The LLM's generated text response
        
    Raises:
        LLMError: If the request fails after all retries
    """
    
    # Build payload for Modal endpoint
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": temperature
    }
    
    # Add system prompt if provided
    if system_message:
        payload["system_prompt"] = system_message
    
    # Log the request
    logger.info(f"Calling LLM with prompt length: {len(prompt)}")
    if metadata:
        logger.debug(f"Metadata: {metadata}")
        
    # Explicitly add prompts to LangSmith trace for visibility
    try:
        run_tree = get_current_run_tree()
        if run_tree:
            # Add full prompts to metadata/extra so they are easily accessible
            run_tree.add_metadata({
                "llm_inputs": {
                    "system_prompt": system_message,
                    "user_prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            })
    except Exception as e:
        logger.debug(f"Could not add metadata to LangSmith: {e}")
    
    # Use a new client for each request to ensure thread safety and loop compatibility
    # Optimization: In a long-running app, we should use a singleton client, 
    # but we need to handle event loops carefully.
    # For now, we'll stick to creating a client but with a longer timeout.
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.post(
                    QWEN_ENDPOINT,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                
                # Validate response structure - Modal returns "response" key
                if "response" not in result:
                    raise LLMError(f"Invalid response format: missing 'response' key. Got: {result}")
                
                text = result["response"]
                logger.info(f"LLM response received (length: {len(text)})")
                
                # Extract token usage and add to LangSmith trace
                usage = result.get("usage", {})
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    
                    logger.info(f"Token usage - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}")
                    
                    # Add token usage to LangSmith run
                    try:
                        run_tree = get_current_run_tree()
                        if run_tree:
                            run_tree.end(
                                outputs={"response": text},
                                extra={
                                    "usage_metadata": {
                                        "input_tokens": prompt_tokens,
                                        "output_tokens": completion_tokens,
                                        "total_tokens": total_tokens,
                                    }
                                }
                            )
                    except Exception as e:
                        logger.debug(f"Could not add token usage to LangSmith: {e}")
                
                return text
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error on attempt {attempt}/{MAX_RETRIES}: {e.response.status_code}")
                if attempt == MAX_RETRIES:
                    raise LLMError(f"HTTP error after {MAX_RETRIES} retries: {e}")
                await asyncio.sleep(RETRY_BACKOFF ** attempt)
                
            except httpx.RequestError as e:
                logger.error(f"Request error on attempt {attempt}/{MAX_RETRIES}: {e}")
                if attempt == MAX_RETRIES:
                    raise LLMError(f"Request error after {MAX_RETRIES} retries: {e}")
                await asyncio.sleep(RETRY_BACKOFF ** attempt)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error on attempt {attempt}/{MAX_RETRIES}: {e}")
                if attempt == MAX_RETRIES:
                    raise LLMError(f"Invalid JSON response after {MAX_RETRIES} retries")
                await asyncio.sleep(RETRY_BACKOFF ** attempt)
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt}/{MAX_RETRIES}: {e}")
                if attempt == MAX_RETRIES:
                    raise LLMError(f"Unexpected error after {MAX_RETRIES} retries: {e}")
                await asyncio.sleep(RETRY_BACKOFF ** attempt)


@traceable(name="call_llm_sync", run_type="llm")
def call_llm_sync(
    prompt: str,
    system_message: str,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    timeout: float = DEFAULT_TIMEOUT,
    metadata: Optional[Dict[str, Any]] = None,
    keep_warm=1
) -> str:
    """
    Synchronous version of call_llm for non-async contexts.
    
    Args:
        prompt: The user prompt/question
        system_message: The system prompt that defines the LLM's behavior
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        metadata: Optional metadata for tracing/logging
        
    Returns:
        str: The LLM's generated text response
        
    Raises:
        LLMError: If the request fails after all retries
    """
    import asyncio
    
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in an async context, raise an error
            raise LLMError("Use async call_llm() in async contexts")
        return loop.run_until_complete(
            call_llm(prompt, system_message, temperature, max_tokens, timeout, metadata)
        )
    except RuntimeError:
        # No event loop exists, create a new one
        return asyncio.run(
            call_llm(prompt, system_message, temperature, max_tokens, timeout, metadata)
        )


def parse_json_response(text: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Safely parse JSON from LLM response with error handling.
    
    The LLM may return JSON wrapped in markdown code blocks or with extra text.
    This function attempts to extract and parse the JSON robustly.
    
    Args:
        text: Raw text response from LLM
        fallback: Default value to return if parsing fails
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        LLMError: If parsing fails and no fallback provided
    """
    try:
        # Try direct parsing first
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    import re
    
    # Pattern for ```json ... ``` or ``` ... ```
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, text, re.DOTALL)
    
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find any JSON object in the text
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(brace_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # All parsing attempts failed
    logger.error(f"Failed to parse JSON from LLM response: {text[:200]}...")
    
    if fallback is not None:
        logger.warning("Using fallback value")
        return fallback
    
    raise LLMError(f"Could not parse JSON from LLM response: {text[:200]}...")
