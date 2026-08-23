import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


class TraceEventType:
    """Standardized lifecycle event types for structured agent execution trace."""
    TURN_STARTED = "TURN_STARTED"
    ACTION_PLANNED = "ACTION_PLANNED"
    ACTION_EXECUTED = "ACTION_EXECUTED"
    OBSERVATION_RECORDED = "OBSERVATION_RECORDED"
    TURN_COMPLETED = "TURN_COMPLETED"
    HANDOFF = "HANDOFF"
    ITERATION_LIMIT_EXHAUSTED = "ITERATION_LIMIT_EXHAUSTED"

    FORBIDDEN_FIELDS = {
        "customer", "internal", "email", "address", "risk_score",
        "warehouse_note", "api_key", "secret", "credentials"
    }


@dataclass
class TraceEvent:
    """
    Lightweight, structured trace event capturing lifecycle execution steps.
    Strictly excludes customer PII, internal notes, and credentials.
    """
    event_type: str
    iteration: int = 0
    action_type: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    success: bool = True
    summary: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace event to dict representation with PII sanitization."""
        data = asdict(self)
        if data.get("parameters") and isinstance(data["parameters"], dict):
            clean_params = {}
            for k, v in data["parameters"].items():
                if k.lower() not in TraceEventType.FORBIDDEN_FIELDS:
                    clean_params[k] = v
            data["parameters"] = clean_params
        return data
