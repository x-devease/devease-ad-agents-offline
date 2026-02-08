"""Negative signal detection for loss aversion outreach."""

from typing import Optional, List
from datetime import datetime, timedelta
from loguru import logger

from models import ShopifyStore, MetaAdData


class NegativeSignal:
    """Detected negative signal."""

    signal_type: str
    severity: str  # critical, high, medium
    description: str
    estimated_loss: float  # Daily estimated loss in USD
    evidence: dict
    recommendation: str


class SignalDetector:
    """Detect negative signals in lead data."""

    def __init__(self):
        self.signals = []

    def detect_all(self, store: ShopifyStore, meta: MetaAdData) -> List[NegativeSignal]:
        """Run all detection checks."""
        signals = []

        # Check 1: Low SKU count but high ad spend
        signal1 = self._check_low_sku_high_ads(store, meta)
        if signal1:
            signals.append(signal1)

        # Check 2: Stale creative (inferred from ad patterns)
        signal2 = self._check_stale_creative(store, meta)
        if signal2:
            signals.append(signal2)

        # Check 3: Low product velocity with active ads
        signal3 = self._check_low_velocity_with_ads(store, meta)
        if signal3:
            signals.append(signal3)

        # Check 4: High avg price but no product variety
        signal4 = self._check_pricing_mismatch(store)
        if signal4:
            signals.append(signal4)

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2}
        signals.sort(key=lambda s: severity_order.get(s.severity, 3))

        return signals

    def _check_low_sku_high_ads(
        self,
        store: ShopifyStore,
        meta: MetaAdData
    ) -> Optional[NegativeSignal]:
        """Low product count but spending on ads."""

        if store.product_count < 15 and meta.ad_count > 5:
            # Estimated loss: burning money on limited catalog
            estimated_loss = meta.ad_count * 50  # $50 per ad per day

            return NegativeSignal(
                signal_type="low_sku_high_ads",
                severity="high",
                description=(
                    f"Burning cash on {meta.ad_count}+ active ads "
                    f"but only {store.product_count} products to convert to"
                ),
                estimated_loss=estimated_loss,
                evidence={
                    "sku_count": store.product_count,
                    "ad_count": meta.ad_count,
                    "threshold_sku": 15,
                    "threshold_ads": 5
                },
                recommendation=(
                    "Expand catalog or pause underperforming ads. "
                    "Each ad needs 20+ SKUs to properly A/B test."
                )
            )

        return None

    def _check_stale_creative(
        self,
        store: ShopifyStore,
        meta: MetaAdData
    ) -> Optional[NegativeSignal]:
        """Check for stale creative based on product launch velocity."""

        # If they have many ads but no new products, creative is likely stale
        if meta.ad_count > 10 and store.launch_velocity_30d < 3:
            estimated_loss = meta.ad_count * 80  # $80 per stale ad per day

            return NegativeSignal(
                signal_type="stale_creative",
                severity="medium",
                description=(
                    f"{meta.ad_count} active ads detected, "
                    f"but only {store.launch_velocity_30d} new products in 30 days. "
                    f"Creative fatigue likely killing ROAS."
                ),
                estimated_loss=estimated_loss,
                evidence={
                    "ad_count": meta.ad_count,
                    "velocity_30d": store.launch_velocity_30d,
                    "last_product_launch": (
                        datetime.now() - timedelta(days=30)
                        if store.launch_velocity_30d == 0 else None
                    )
                },
                recommendation=(
                    "Launch 3-5 new products this week. "
                    "Fresh creative = 3x ROAS bump typically."
                )
            )

        return None

    def _check_low_velocity_with_ads(
        self,
        store: ShopifyStore,
        meta: MetaAdData
    ) -> Optional[NegativeSignal]:
        """Active ads but zero product launches."""

        if meta.active_ads and store.launch_velocity_30d == 0:
            estimated_loss = 200  # $200/day from missed growth

            return NegativeSignal(
                signal_type="zero_velocity_with_ads",
                severity="critical",
                description=(
                    f"You're spending on ads but haven't launched "
                    f"a single new product in 30+ days. "
                    f"Your competitors are eating your market share."
                ),
                estimated_loss=estimated_loss,
                evidence={
                    "active_ads": meta.active_ads,
                    "velocity_30d": store.launch_velocity_30d,
                    "days_since_launch": 30
                },
                recommendation=(
                    "Immediate action needed: Launch 1-2 new products this week. "
                    "Ad spend without new offers = diminishing returns."
                )
            )

        return None

    def _check_pricing_mismatch(
        self,
        store: ShopifyStore
    ) -> Optional[NegativeSignal]:
        """High average price but low product count."""

        if store.avg_price > 100 and store.product_count < 20:
            estimated_loss = store.product_count * 5  # $5 per SKU per day

            return NegativeSignal(
                signal_type="pricing_mismatch",
                severity="medium",
                description=(
                    f"${store.avg_price:.0f} average price point "
                    f"but only {store.product_count} SKUs. "
                    f"High-ticket needs catalog depth for trust."
                ),
                estimated_loss=estimated_loss,
                evidence={
                    "avg_price": store.avg_price,
                    "sku_count": store.product_count,
                    "price_threshold": 100,
                    "sku_threshold": 20
                },
                recommendation=(
                    "Add 10-15 SKUs at lower price points. "
                    "Entry products build trust for high-ticket core."
                )
            )

        return None


def format_signal_for_outreach(signal: NegativeSignal) -> str:
    """Format signal for outreach message."""

    emoji_map = {
        "critical": "🚨",
        "high": "⚠️",
        "medium": "💡"
    }

    emoji = emoji_map.get(signal.severity, "")

    return (
        f"{emoji} {signal.description}\n\n"
        f"💸 Estimated daily loss: ${signal.estimated_loss:,.0f}\n"
        f"🔧 Fix: {signal.recommendation}"
    )
