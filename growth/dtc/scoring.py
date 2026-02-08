"""Lead intent scoring logic."""

from typing import Optional
from loguru import logger

from models import Lead, ShopifyStore, MetaAdData, ContactInfo


class IntentScorer:
    """Score leads based on multiple signals."""

    def __init__(
        self,
        weights: dict = None,
        min_sku: int = 10,
        min_ads: int = 3
    ):
        self.weights = weights or {
            "sku_count": 0.25,
            "launch_velocity": 0.30,
            "ad_count": 0.25,
            "ad_recency": 0.20
        }
        self.min_sku = min_sku
        self.min_ads = min_ads

    def score_lead(self, lead: Lead) -> float:
        """Calculate intent score for a lead."""
        score = 0.0

        store = lead.store
        meta = lead.meta_ads

        if not store or not meta:
            logger.warning(f"Missing data for {lead.domain}")
            return 0.0

        # Factor 1: SKU count (0-25 points)
        sku_score = min(store.product_count / self.min_sku, 25)
        score += sku_score * self.weights["sku_count"]
        logger.debug(f"{lead.domain}: SKU score = {sku_score:.1f}")

        # Factor 2: Launch velocity (0-35 points)
        velocity_score = min(store.launch_velocity_30d * 3, 35)
        score += velocity_score * self.weights["launch_velocity"]
        logger.debug(f"{lead.domain}: Velocity score = {velocity_score:.1f}")

        # Factor 3: Ad count (0-30 points)
        ad_score = min(meta.ad_count * 2, 30)
        score += ad_score * self.weights["ad_count"]
        logger.debug(f"{lead.domain}: Ad score = {ad_score:.1f}")

        # Factor 4: Ad recency/active (0-10 points)
        recency_score = 0
        if meta.active_ads:
            recency_score += 5
        if getattr(meta, "has_recent_activity", False):
            recency_score += 5
        score += recency_score * self.weights["ad_recency"]
        logger.debug(f"{lead.domain}: Recency score = {recency_score:.1f}")

        # Bonus: Has contact info (+5 points)
        if lead.contact and (lead.contact.email or lead.contact.twitter_handle):
            score += 5
            logger.debug(f"{lead.domain}: Contact bonus = +5")

        final_score = round(min(score, 100), 2)
        logger.success(f"{lead.domain}: Final score = {final_score}")

        return final_score

    def rank_leads(self, leads: list[Lead]) -> list[Lead]:
        """Sort leads by intent score."""
        for lead in leads:
            lead.intent_score = self.score_lead(lead)

        ranked = sorted(leads, key=lambda x: x.intent_score, reverse=True)

        logger.info(f"Ranked {len(ranked)} leads")
        for i, lead in enumerate(ranked[:10], 1):
            logger.info(f"  {i}. {lead.domain}: {lead.intent_score}")

        return ranked

    def filter_high_intent(
        self,
        leads: list[Lead],
        min_score: float = 50.0
    ) -> list[Lead]:
        """Filter leads above score threshold."""
        high_intent = [l for l in leads if l.intent_score >= min_score]
        logger.info(f"Filtered {len(high_intent)}/{len(leads)} high-intent leads")
        return high_intent
