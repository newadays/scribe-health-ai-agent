"""Jev client (TypeSafe System One), the decision-model counterpart of agents/llm.py.

Jev does not generate text. It answers typed questions (Choice, Noul, Score)
about a piece of text and returns probabilities and a confidence for each.
"""

from functools import lru_cache
from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient

load_dotenv()


@lru_cache(maxsize=1)
def get_jev() -> TypeSafeClient:
    """Create the client on first use.

    TypeSafeClient() reads TYPESAFE_API_KEY from the environment and raises
    TypeSafeError if it is missing, so we build it lazily instead of at import
    time. The model defaults to "jev-latest".
    """
    return TypeSafeClient()
