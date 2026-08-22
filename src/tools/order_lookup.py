import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass
class CustomerSafeItem:
    """Safe item structure excluding internal SKUs or cost fields."""
    name: str
    quantity: int
    final_sale: bool


@dataclass
class CustomerSafeOrderResult:
    """
    Sanitized, customer-safe order lookup result.
    Strictly omits customer PII (name, email, address) and internal notes/scores.
    """
    found: bool
    order_id: Optional[str] = None
    membership_tier: Optional[str] = None
    placed_at: Optional[str] = None
    status: Optional[str] = None  # Authoritative status
    status_updated_at: Optional[str] = None
    shipped_at: Optional[str] = None
    delivered_at: Optional[str] = None
    carrier: Optional[str] = None
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[str] = None
    customer_safe_message: Optional[str] = None
    items: List[CustomerSafeItem] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary format suitable for LLM context."""
        return asdict(self)


class OrderLookupTool:
    """
    Secure order lookup tool reading from data/orders.json.
    Acts as a security boundary: sanitizes database records before exposing to LLM/user.
    """

    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            data_path = Path("data/orders.json")
        self.data_path = data_path
        self._orders_map: Dict[str, Dict[str, Any]] = {}
        self.snapshot_at: Optional[str] = None
        self._load_data()

    def _load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Orders dataset file not found at {self.data_path}")

        content = self.data_path.read_text(encoding="utf-8")
        data = json.loads(content)
        self.snapshot_at = data.get("snapshot_at")

        for order in data.get("orders", []):
            order_id = order.get("order_id", "").strip().upper()
            if order_id:
                self._orders_map[order_id] = order

    @staticmethod
    def normalize_order_id(raw_id: str) -> str:
        """Normalize user order ID (uppercase, strip whitespace, remove surrounding punctuation)."""
        if not raw_id:
            return ""
        cleaned = raw_id.strip()
        cleaned = re.sub(r"^[^\w\-]+|[^\w\-]+$", "", cleaned)
        return cleaned.upper()

    def lookup(self, raw_order_id: str) -> CustomerSafeOrderResult:
        """
        Execute order lookup and return a customer-safe sanitized result.
        """
        order_id = self.normalize_order_id(raw_order_id)

        if not order_id or order_id not in self._orders_map:
            return CustomerSafeOrderResult(
                found=False,
                order_id=order_id if order_id else raw_order_id,
                error_message=f"Order '{raw_order_id}' was not found. Please check the order ID or contact support."
            )

        raw_order = self._orders_map[order_id]
        status = raw_order.get("status")

        # Handle ETA sanitization for cancelled or returned orders
        estimated_delivery = raw_order.get("estimated_delivery")
        if status in ("cancelled", "returned"):
            # Operational systems may retain stale ETA; do not present stale ETA as arriving
            estimated_delivery = None

        # Build customer-safe items list
        safe_items = []
        for item in raw_order.get("items", []):
            safe_items.append(
                CustomerSafeItem(
                    name=item.get("name", "Unknown Item"),
                    quantity=item.get("quantity", 1),
                    final_sale=bool(item.get("final_sale", False)),
                )
            )

        # Build CustomerSafeOrderResult - strictly excluding customer and internal objects
        return CustomerSafeOrderResult(
            found=True,
            order_id=raw_order.get("order_id"),
            membership_tier=raw_order.get("membership_tier"),
            placed_at=raw_order.get("placed_at"),
            status=status,
            status_updated_at=raw_order.get("status_updated_at"),
            shipped_at=raw_order.get("shipped_at"),
            delivered_at=raw_order.get("delivered_at"),
            carrier=raw_order.get("carrier"),
            tracking_number=raw_order.get("tracking_number"),
            estimated_delivery=estimated_delivery,
            customer_safe_message=raw_order.get("customer_safe_message"),
            items=safe_items,
            error_message=None,
        )
