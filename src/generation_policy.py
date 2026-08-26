from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class GroundedGenerationPolicy:
    """
    Deterministic configuration object defining the grounded generation contract for Aster & Row.

    Contract Rules:
    - require_evidence: Factual claims must be backed by authoritative retrieved evidence or sanitized order data.
    - prohibit_unsupported_claims: Models must not invent policies, shipping rules, or warranty terms.
    - require_source_citations: Customer-facing policy claims must include inline source citations.
    - allow_abstention: When evidence is insufficient, the system abstains and escalates to human support.
    - prefer_escalation_on_conflict: Conflicting active policy documents trigger human escalation rather than arbitrary selection.
    """
    require_evidence: bool = True
    prohibit_unsupported_claims: bool = True
    require_source_citations: bool = True
    allow_abstention: bool = True
    prefer_escalation_on_conflict: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "require_evidence": self.require_evidence,
            "prohibit_unsupported_claims": self.prohibit_unsupported_claims,
            "require_source_citations": self.require_source_citations,
            "allow_abstention": self.allow_abstention,
            "prefer_escalation_on_conflict": self.prefer_escalation_on_conflict,
        }
