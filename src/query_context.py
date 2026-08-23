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

        # If current query is already long and self-contained, or identical, avoid over-expanding
        if cleaned_query.lower() in previous_user_q.lower():
            return cleaned_query

        # Combine previous user context topic with current query for vector search
        return f"{previous_user_q} {cleaned_query}"
