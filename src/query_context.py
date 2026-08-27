from typing import List, Optional
from src.memory import ConversationTurn


class QueryContextualizer:
    """
    Constructs a retrieval-oriented query from the current user query
    and bounded conversation context.

    Separates the responsibility of retrieval query expansion from:
    1. Raw user question preservation in final LLM prompt payload.
    2. Session memory storage.
    3. Grounded prompt context building.
    """

    @staticmethod
    def build_retrieval_query(
        user_query: str,
        history_turns: Optional[List[ConversationTurn]] = None
    ) -> str:
        """
        Builds a contextualized retrieval query string for vector search.

        Args:
            user_query: The raw user input for the current turn.
            history_turns: Bounded list of previous ConversationTurn objects for this session.

        Returns:
            A string optimized for vector search embedding.
        """
        cleaned_query = user_query.strip()
        if not history_turns:
            return cleaned_query

        # Use the most recent previous user turn to contextualize ambiguous follow-ups
        last_turn = history_turns[-1]
        previous_user_q = last_turn.user_query.strip()

        # If current query is already identical or contained, return clean query
        if cleaned_query.lower() in previous_user_q.lower():
            return cleaned_query

        query_lower = cleaned_query.lower()
        words = query_lower.split()

        # Check if current query is a short/ambiguous follow-up (e.g. "What about Canada?", "When will it arrive?", "Is it free?")
        is_followup = (
            any(query_lower.startswith(prefix) for prefix in ["what about", "how about", "what of", "is it", "does it", "can i", "what if", "and ", "also ", "why "])
            or any(w in ("it", "that", "this", "they", "them", "there") for w in words)
            or len(words) <= 5
        )

        # Standalone, self-contained questions (> 5 words without follow-up pronouns/phrases) do not need historical query expansion
        if not is_followup:
            return cleaned_query

        # Combine previous user context topic with current query for vector search
        return f"{previous_user_q} {cleaned_query}"
