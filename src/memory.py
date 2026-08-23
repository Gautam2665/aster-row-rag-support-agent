from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time


@dataclass
class ConversationTurn:
    """Represents a single user-assistant interaction turn."""
    user_query: str
    assistant_response: str
    timestamp: float = field(default_factory=time.time)


class ConversationMemory:
    """
    Bounded in-memory conversation history for a single session.
    Retains up to max_turns turns in a FIFO queue.
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []

    def add_turn(self, user_query: str, assistant_response: str):
        """Add a completed turn, evicting the oldest turn if max_turns is exceeded."""
        turn = ConversationTurn(
            user_query=user_query.strip(),
            assistant_response=assistant_response.strip()
        )
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def get_recent_turns(self) -> List[ConversationTurn]:
        """Return all retained turns in chronological order."""
        return list(self.turns)

    def clear(self):
        """Clear all stored turns in this memory session."""
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns)


class SessionMemoryStore:
    """
    Session manager mapping session_id -> ConversationMemory.
    Guarantees session isolation so history does not leak across sessions.
    """

    def __init__(self, max_turns_per_session: int = 5):
        self.max_turns_per_session = max_turns_per_session
        self._sessions: Dict[str, ConversationMemory] = {}

    def get_memory(self, session_id: str) -> ConversationMemory:
        """Get or create the ConversationMemory instance for a session_id."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(max_turns=self.max_turns_per_session)
        return self._sessions[session_id]

    def add_turn(self, session_id: str, user_query: str, assistant_response: str):
        """Add a turn to a specific session_id."""
        memory = self.get_memory(session_id)
        memory.add_turn(user_query, assistant_response)

    def get_recent_turns(self, session_id: str) -> List[ConversationTurn]:
        """Get recent turns for a specific session_id."""
        return self.get_memory(session_id).get_recent_turns()

    def clear_session(self, session_id: str):
        """Clear memory for a specific session_id."""
        if session_id in self._sessions:
            self._sessions[session_id].clear()

    def clear_all(self):
        """Clear memory across all sessions."""
        self._sessions.clear()
