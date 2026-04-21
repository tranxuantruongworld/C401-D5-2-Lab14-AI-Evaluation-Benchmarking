import asyncio
import random
from functools import wraps

def retry_with_exponential_backoff(
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    max_retries: int = 5,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """Decorator for retrying async functions with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        retries += 1
                        if retries > max_retries:
                            print(f"Max retries reached for {func.__name__}. Error: {e}")
                            raise e
                        
                        delay = min(base_delay * (exponential_base ** (retries - 1)), max_delay)
                        if jitter:
                            delay *= (0.5 + random.random())
                        
                        print(f"Rate limit hit in {func.__name__}. Retrying in {delay:.2f}s... (Attempt {retries}/{max_retries})")
                        await asyncio.sleep(delay)
                    else:
                        # For other exceptions, we might want to re-raise immediately
                        raise e
        return wrapper
    return decorator
