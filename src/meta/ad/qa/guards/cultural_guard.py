"""
Cultural Guard - Compliance safety validation.

This guard checks for regional cultural taboos and compliance issues.
"""

import time
from typing import List, Optional

from ..schemas.audit_report import GuardStatus, CulturalResult, RiskLevel
from ..vlms.base import VLMClient


# Regional taboo libraries
REGION_RULES = {
    "Middle_East": {
        "taboos": [
            # Religious/Islamic taboos
            "exposed skin (short skirts, swimwear, revealing tops, tight clothing)",
            "pork products (ham, bacon, pork meat, lard)",
            "alcohol beverages (beer, wine, liquor, cocktail imagery)",
            "religious symbols (crosses, stars, sacred text, religious iconography)",
            "mosques or religious sites used for commercial purposes",

            # Cultural norms
            "left hand used for eating, greeting, or handing objects",
            "foot soles visible or pointed at people/objects (offensive)",
            "men and women in intimate or touching proximity",
            "women without headscarves in conservative contexts",
            "gender-mixing in inappropriate contexts",

            # Regional sensitivities
            "Israeli flags or Hebrew text in certain markets",
            "political maps or territorial disputes",
            "images of prophets or religious figures"
        ],
        "context": "Conservative Islamic values and cultural norms. High sensitivity to religious and gender norms.",
        "recommendations": [
            "Use modest clothing standards",
            "Avoid physical contact between genders in imagery",
            "Respect religious imagery guidelines",
            "Be mindful of hand gestures (use right hand)",
            "Consider local market requirements for gender segregation"
        ]
    },

    "East_Asia": {
        "taboos": [
            # Aesthetic preferences
            "excessive color clashing or neon saturation",
            "cluttered chaotic layouts",
            "dark or overly somber color schemes",

            # Cultural superstitions
            "number 4 visible (considered unlucky in China/Korea - sounds like 'death')",
            "number 9 visible in Japan (associated with suffering)",
            "white chrysanthemum flowers (funeral association)",
            "lotus position (funeral association in some regions)",

            # Social norms
            "models with non-East Asian facial features",
            "direct eye contact with authority figures (can be seen as disrespectful)",
            "thumbs down gesture (offensive in some contexts)",
            "pointing with index finger (use open hand)",
            "showing bottom of shoes (offensive)",

            # Business culture
            "overt confrontation or aggression",
            "excessive individualism over group harmony"
        ],
        "context": "Minimalist aesthetics, cultural resonance, harmony. High importance placed on subtlety and group cohesion.",
        "recommendations": [
            "Use muted, harmonious color palettes",
            "Maintain clean, organized layouts",
            "Use locally relevant models and imagery",
            "Avoid unlucky numbers in pricing or displays",
            "Emphasize harmony and collectivism",
            "Respect hierarchical relationships in imagery"
        ]
    },

    "USA_Europe": {
        "taboos": [
            # DEI and discrimination
            "racial stereotypes or caricatures",
            "gender role stereotypes (e.g., only women in kitchen, only men in leadership)",
            "age discrimination (excluding older adults)",
            "disability insensitivity",
            "LGBTQ+ stereotypes or lack of representation",

            # Legal compliance
            "misleading claims or false promises",
            "unrealistic before/after results",
            "fine print that contradicts main message",
            "discriminatory language or imagery",

            # Cultural sensitivity
            "cultural appropriation",
            "exploitation of sensitive groups",
            "tragedy exploitation or disaster insensitivity",
            "political polarization or divisive content",

            # Privacy and consent
            "images of people without model releases",
            "children in inappropriate contexts",
            "personal data or private information visible"
        ],
        "context": "DEI standards, truth in advertising, inclusivity, and strict regulatory compliance.",
        "recommendations": [
            "Ensure diverse and representative casting",
            "Avoid harmful stereotypes",
            "Make only substantiated and verifiable claims",
            "Consider accessibility guidelines (WCAG)",
            "Include model releases for all identifiable people",
            "Review against FTC/ASA advertising guidelines",
            "Test with diverse focus groups"
        ]
    },

    "India": {
        "taboos": [
            # Religious sensitivities
            "beef products (sacred in Hinduism)",
            "cow imagery in disrespectful contexts",
            "religious symbols used commercially (Om, swastika, etc.)",
            "gods or deities in commercial contexts",
            "religious insensitivity (Hindu/Muslim/Sikh symbols)",

            # Social norms
            "revealing or provocative clothing",
            "public displays of affection (kissing, touching)",
            "foot pointing at people or religious objects",
            "touching someone's head (considered sacred)",

            # Cultural considerations
            "caste-based imagery or references",
            "political or religious controversies",
            "alcohol in conservative regions",
            "non-vegetarian food in vegetarian contexts"
        ],
        "context": "Hindu and Muslim sensitivities, conservative values, diverse religious landscape.",
        "recommendations": [
            "Respect religious diversity and symbols",
            "Use modest dress standards",
            "Avoid religious iconography in commercial contexts",
            "Be mindful of regional dietary preferences (veg/non-veg)",
            "Consider linguistic diversity (many languages)",
            "Avoid political or religious controversies",
            "Show family values and community connections"
        ]
    },

    "Latin_America": {
        "taboos": [
            # Religious sensitivities
            "disrespect to Catholic symbols or imagery",
            "irreverent use of religious figures",

            # Cultural norms
            "negative stereotypes of Latin American people",
            "misrepresentation of local customs",
            "classist imagery (extreme wealth vs poverty)",

            # Historical sensitivities
            "references to colonialism or conquest",
            "political polarization content",
            "drug trade or cartel imagery",

            # Social considerations
            "objectification of women (machismo culture)",
            "violence or weapons imagery"
        ],
        "context": "Catholic traditions, family-oriented culture, sensitivity to class and historical issues.",
        "recommendations": [
            "Emphasize family values and community",
            "Show vibrant, warm color palettes",
            "Use diverse, authentic representation",
            "Avoid negative cultural stereotypes",
            "Include family gatherings and social connections",
            "Be mindful of religious holidays and traditions"
        ]
    },

    "Southeast_Asia": {
        "taboos": [
            # Religious diversity
            "disrespect to Buddhist symbols or imagery",
            "insensitivity to Muslim customs (in Malaysia, Indonesia)",
            "disrespect to Hindu traditions (in Bali, Singapore)",

            # Cultural norms
            "touching someone's head (considered sacred)",
            "pointing with feet or showing soles",
            "public displays of affection",
            "raising voice or showing anger (loss of face)",

            # Royal sensitivity (especially Thailand)
            "disrespect to monarchy or royal family",
            "misuse of national symbols",

            # Superstitions
            "unlucky numbers or dates"
        ],
        "context": "Buddhist, Muslim, and Hindu influences. High importance on respect and 'saving face'.",
        "recommendations": [
            "Maintain respectful, harmonious imagery",
            "Emphasize community and togetherness",
            "Avoid confrontation or aggression",
            "Respect religious diversity in imagery",
            "Be mindful of royal sensitivities (especially Thailand)",
            "Use warm, inviting visual styles"
        ]
    },

    "Africa": {
        "taboos": [
            # Colonial sensitivity
            "colonial-era imagery or references",
            "stereotypical 'tribal' imagery",
            "misrepresentation of traditional practices",

            # Regional diversity
            "ignoring country-specific cultural norms",
            "negative stereotypes of African people",
            "poverty porn or exploitation imagery",

            # Religious considerations
            "disrespect to Islamic practices (North Africa)",
            "disrespect to Christian traditions (Sub-Saharan)",
            "disrespect to traditional African religions",

            # Social norms
            "elders shown in disrespectful contexts",
            "inappropriate dress standards for conservative regions"
        ],
        "context": "Extreme regional and cultural diversity. Colonial history and religious traditions shape sensitivities.",
        "recommendations": [
            "Show modern, progressive Africa (not just stereotypes)",
            "Respect country-specific cultural differences",
            "Avoid poverty exploitation imagery",
            "Include diverse representation across the continent",
            "Be mindful of religious diversity",
            "Emphasize progress and development",
            "Respect traditional values while showing modernity"
        ]
    },

    "Russia_Eastern_Europe": {
        "taboos": [
            # Political sensitivities
            "political controversy or criticism",
            "LGBTQ+ imagery (restrictions in some markets)",

            # Cultural norms
            "excessive cheeriness or enthusiasm (seen as insincere)",
            "weak or passive protagonists",

            # Historical sensitivities
            "Soviet era references",
            "WWII imagery without proper respect"
        ],
        "context": "Conservative social values, political sensitivities, preference for strength and directness.",
        "recommendations": [
            "Use serious, straightforward imagery",
            "Emphasize strength and quality",
            "Avoid overly casual or playful tones",
            "Be mindful of political sensitivities",
            "Show competence and reliability"
        ]
    },

    "Global": {
        "taboos": [
            # Universal safety
            "violence or weapons (guns, knives, combat)",
            "illegal activities or criminal behavior",
            "hate speech or hate symbols (swastikas, etc.)",
            "self-harm or suicide references",
            "sexual content or nudity",

            # Platform policies
            "content that violates platform terms of service",
            "copyrighted material without permission",
            "trademark infringement"
        ],
        "context": "Universal safety standards and platform policy compliance.",
        "recommendations": [
            "Follow platform content policies",
            "Obtain legal review for regulated industries",
            "Ensure all imagery has proper rights clearance",
            "Review against hate speech policies",
            "Avoid controversial or polarizing content"
        ]
    }
}


class CulturalGuard:
    """
    Validates cultural compliance of generated images.

    Checks for region-specific taboos and compliance issues.
    """

    def __init__(
        self,
        vlm_client: VLMClient,
        target_region: str = "Global",
        risk_threshold: str = "HIGH",
        custom_rules: Optional[List[str]] = None
    ):
        """
        Initialize CulturalGuard.

        Args:
            vlm_client: VLM client for image analysis
            target_region: Target region for compliance check
            risk_threshold: Risk level that triggers FAIL ("HIGH", "MEDIUM", "LOW")
            custom_rules: Additional customer-specific taboos
        """
        self.vlm_client = vlm_client
        self.target_region = target_region
        self.risk_threshold = risk_threshold
        self.custom_rules = custom_rules or []

        # Get regional taboos
        region_config = REGION_RULES.get(target_region, REGION_RULES["Global"])
        self.taboos = region_config["taboos"]
        self.context = region_config["context"]

        # Add custom rules
        if self.custom_rules:
            self.taboos.extend(self.custom_rules)

    def check(self, image_path: str) -> CulturalResult:
        """
        Check cultural compliance of an image.

        Args:
            image_path: Path to generated image

        Returns:
            CulturalResult with validation status
        """
        start_time = time.time()

        try:
            # Use VLM client's built-in method
            result = self.vlm_client.check_culture(
                image_path=image_path,
                region=self.target_region,
                taboos=self.taboos
            )

            execution_time = (time.time() - start_time) * 1000

            # Extract risk and issues
            risk_level_str = result.get("risk_level", "LOW")
            risk_level = RiskLevel(risk_level_str.lower())
            detected_issues = result.get("detected_issues", [])
            confidence = result.get("confidence", 0.0)
            reasoning = result.get("reasoning", "")

            # Determine status based on risk threshold
            if self._should_fail(risk_level):
                status = GuardStatus.FAIL
                fail_reason = f"Cultural compliance risk: {risk_level_str}. Issues: {detected_issues}"
            else:
                status = GuardStatus.PASS
                fail_reason = ""

            # Build reasoning
            if status == GuardStatus.PASS:
                reasoning = f"Cultural compliance PASSED for region {self.target_region}. Risk level: {risk_level_str}"
                if detected_issues:
                    reasoning += f". Minor concerns: {', '.join(detected_issues[:2])}"
            else:
                reasoning = fail_reason
                if reasoning:
                    reasoning += f". {reasoning}"

            return CulturalResult(
                guard_name="cultural",
                status=status,
                reasoning=reasoning,
                metrics={
                    "target_region": self.target_region,
                    "risk_level": risk_level_str,
                    "risk_threshold": self.risk_threshold,
                    "detected_issues": detected_issues,
                    "confidence": confidence,
                    "taboos_checked": len(self.taboos)
                },
                execution_time_ms=execution_time,
                risk_level=risk_level,
                detected_issues=detected_issues,
                confidence=confidence
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return CulturalResult(
                guard_name="cultural",
                status=GuardStatus.FAIL,
                reasoning=f"Error during cultural check: {str(e)}",
                metrics={"error": str(e)},
                execution_time_ms=execution_time,
                risk_level=RiskLevel.LOW,
                detected_issues=[],
                confidence=0.0
            )

    def _should_fail(self, risk_level: RiskLevel) -> bool:
        """
        Determine if risk level should trigger FAIL.

        Args:
            risk_level: Detected risk level

        Returns:
            True if should fail
        """
        if self.risk_threshold == "HIGH":
            return risk_level == RiskLevel.HIGH
        elif self.risk_threshold == "MEDIUM":
            return risk_level in [RiskLevel.HIGH, RiskLevel.MEDIUM]
        else:  # LOW threshold - fail on any risk
            return risk_level != RiskLevel.LOW
