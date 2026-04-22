"""
Utility functions for EoH-U framework
"""

import re
import threading
from typing import Tuple, Callable


def extract_thought_and_code(text: str) -> Tuple[str, str]:
    """
    Extract algorithm description and code from LLM output

    Args:
        text: LLM response text

    Returns:
        Tuple of (thought, code)
    """
    # Extract algorithm description
    thought_pattern = re.compile(r'<Algorithm Description>\s*(.*?)\s*<Code>', re.DOTALL)
    thought_match = thought_pattern.search(text)
    thought = thought_match.group(1).strip() if thought_match else ''

    # Extract code
    code_pattern = re.compile(r'<Code>\s*```python\s*(.*?)\s*```', re.DOTALL)
    code_match = code_pattern.search(text)
    code = code_match.group(1).strip() if code_match else ''

    return thought, code


def timeout_decorator(seconds: int):
    """
    Decorator to add timeout to a function

    Args:
        seconds: Timeout in seconds

    Returns:
        Decorated function
    """
    class TimeoutException(Exception):
        pass

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            result = [TimeoutException('Function call timed out')]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                raise TimeoutException('Function call timed out')

            if isinstance(result[0], BaseException):
                raise result[0]

            return result[0]

        return wrapper

    return decorator
