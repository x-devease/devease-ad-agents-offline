# Ad Reviewer - Final Design

**Status:** Ready to implement
**Complexity:** Simple, focused, production-ready
**Files:** 4 Python files + 2 config directories

---

## Overview

The Ad Reviewer evaluates ad creatives across **3 essential dimensions**:

1. **Brand Compliance** - Logo, colors, style
2. **Safety** - Weapons, violence, fraud, adult content, hate, etc.
3. **Quality** - Resolution, sharpness, lighting, composition

**Output:** Score (0-100), Approve/Reject decision, Recommendations

---

## File Structure

```
src/meta/ad/reviewer/
├── __init__.py           # Package exports
├── reviewer.py           # Main AdReviewer class
├── checks.py             # Check functions (brand, safety, quality)
├── prompts.py            # GPT-4 Vision prompts
└── models.py             # ReviewResult dataclass

config/ad/reviewer/
├── brands/               # Brand guidelines
│   ├── moprobo.yaml
│   ├── ecoflow.yaml
│   └── generic.yaml
└── regions/              # Regional risk profiles
    ├── global.yaml
    ├── us.yaml
    ├── uk.yaml
    ├── de.yaml
    ├── jp.yaml
    ├── sa.yaml           # Saudi Arabia
    └── br.yaml           # Brazil

tests/
└── test_reviewer.py      # Unit tests
```

**Total: 4 Python files**

---

## Implementation

### 1. Main Reviewer Class

```python
# src/meta/ad/reviewer/reviewer.py

"""
Ad Reviewer - Simple, focused ad creative evaluation.

Usage:
    from src.meta.ad.reviewer import AdReviewer

    reviewer = AdReviewer(brand="moprobo", region="US")
    result = reviewer.review("creative.jpg")

    if result.approved:
        print(f"✅ {result.score}/100")
    else:
        print(f"❌ {result.recommendations}")
"""

from pathlib import Path
from typing import Dict, Optional
import json
import base64
import logging

from openai import OpenAI

from .models import ReviewResult
from .checks import check_brand, check_safety, check_quality
from .prompts import ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


class AdReviewer:
    """
    Ad creative reviewer.

    Evaluates creatives across brand compliance, safety, and quality.
    """

    def __init__(
        self,
        brand: str,
        region: str = "US",
        api_key: Optional[str] = None
    ):
        """
        Initialize reviewer.

        Args:
            brand: Brand name (must have config in config/ad/reviewer/brands/)
            region: Region code (must have config in config/ad/reviewer/regions/)
            api_key: OpenAI API key (optional, uses env var if not provided)
        """
        self.brand = brand
        self.region = region
        self.client = OpenAI(api_key=api_key)

        # Load configurations
        self.brand_config = self._load_config(f"config/ad/reviewer/brands/{brand}.yaml")
        self.region_config = self._load_config(f"config/ad/reviewer/regions/{region}.yaml")

        logger.info(f"AdReviewer initialized: brand={brand}, region={region}")

    def review(self, image_path: str) -> ReviewResult:
        """
        Review a creative image.

        Args:
            image_path: Path to creative image

        Returns:
            ReviewResult with score, approval, and recommendations
        """
        logger.info(f"Reviewing: {image_path}")

        try:
            # Step 1: Analyze with GPT-4 Vision (single call)
            analysis = self._analyze(image_path)

            # Step 2: Run checks
            brand_result = check_brand(analysis, self.brand_config)
            safety_result = check_safety(analysis, self.region_config)
            quality_result = check_quality(image_path, analysis)

            # Step 3: Calculate overall score
            score = self._calculate_score(brand_result, safety_result, quality_result)

            # Step 4: Determine approval
            approved = self._is_approved(score, brand_result, safety_result, quality_result)

            # Step 5: Generate recommendations
            recommendations = self._generate_recommendations(
                brand_result, safety_result, quality_result
            )

            result = ReviewResult(
                image_path=image_path,
                brand=self.brand,
                region=self.region,
                score=score,
                approved=approved,
                brand_compliance=brand_result,
                safety=safety_result,
                quality=quality_result,
                recommendations=recommendations
            )

            logger.info(f"Review complete: score={score:.0f}, approved={approved}")

            return result

        except Exception as e:
            logger.error(f"Review failed: {e}", exc_info=True)
            raise ReviewError(f"Failed to review {image_path}: {e}")

    def review_batch(
        self,
        image_paths: list[str],
        save_reports: bool = True
    ) -> Dict[str, ReviewResult]:
        """
        Review multiple images.

        Args:
            image_paths: List of image paths
            save_reports: Whether to save individual reports

        Returns:
            Dict mapping image_path -> ReviewResult
        """
        results = {}

        for path in image_paths:
            try:
                result = self.review(path)
                results[path] = result

                if save_reports:
                    report_path = self._get_report_path(path)
                    result.save(report_path)

            except Exception as e:
                logger.error(f"Failed to review {path}: {e}")
                results[path] = None

        return results

    def _analyze(self, image_path: str) -> Dict:
        """Analyze image with GPT-4 Vision."""
        base64_image = self._encode_image(image_path)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ANALYSIS_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    def _calculate_score(
        self,
        brand_result: Dict,
        safety_result: Dict,
        quality_result: Dict
    ) -> float:
        """Calculate overall score (0-100)."""
        brand_score = brand_result['score']
        safety_score = safety_result['score']
        quality_score = quality_result['score']

        # Weighted average
        return (brand_score * 0.40 + safety_score * 0.35 + quality_score * 0.25)

    def _is_approved(
        self,
        score: float,
        brand_result: Dict,
        safety_result: Dict,
        quality_result: Dict
    ) -> bool:
        """Determine if creative is approved."""
        # Minimum score threshold
        if score < 70:
            return False

        # All checks must pass
        if not brand_result['passed']:
            return False

        if not safety_result['passed']:
            return False

        if not quality_result['passed']:
            return False

        # No critical safety issues
        if safety_result.get('critical_issues'):
            return False

        return True

    def _generate_recommendations(
        self,
        brand_result: Dict,
        safety_result: Dict,
        quality_result: Dict
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Brand recommendations
        recommendations.extend(brand_result.get('recommendations', []))

        # Safety recommendations (prioritize these)
        if safety_result.get('recommendations'):
            if safety_result.get('critical_issues'):
                recommendations.insert(
                    0,
                    "CRITICAL: Address safety issues immediately"
                )
            recommendations.extend(safety_result.get('recommendations', []))

        # Quality recommendations
        recommendations.extend(quality_result.get('recommendations', []))

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _load_config(self, path: str) -> Dict:
        """Load YAML configuration."""
        import yaml

        full_path = Path(path)
        if not full_path.exists():
            logger.warning(f"Config not found: {path}, using defaults")
            return {}

        with open(full_path) as f:
            return yaml.safe_load(f)

    def _get_report_path(self, image_path: str) -> str:
        """Generate report path for image."""
        image_path = Path(image_path)
        output_dir = Path("results/ad/reviewer")
        output_dir.mkdir(parents=True, exist_ok=True)

        report_name = f"{image_path.stem}_review.json"
        return str(output_dir / report_name)


class ReviewError(Exception):
    """Review error."""
    pass
```

### 2. Check Functions

```python
# src/meta/ad/reviewer/checks.py

"""
Review check functions.

Each check returns:
{
    'passed': bool,
    'score': float (0-100),
    'issues': list[str],
    'recommendations': list[str]
}
"""

from typing import Dict, List
from PIL import Image


def check_brand(analysis: Dict, config: Dict) -> Dict:
    """
    Check brand compliance.

    Validates:
    - Logo presence, size, placement, quality
    - Color palette match
    - Style/appropriateness
    """
    issues = []
    score = 100.0

    # Logo check
    logo = analysis.get('logo', {})

    if logo.get('present') != 'yes':
        issues.append("Logo not detected")
        score -= 50
    else:
        # Size check
        size_str = logo.get('size_percentage', '0%')
        try:
            size = float(size_str.rstrip('%'))
            min_size = config.get('logo_min_size', 5.0)
            max_size = config.get('logo_max_size', 10.0)

            if size < min_size:
                issues.append(f"Logo too small: {size:.1f}% (minimum: {min_size}%)")
                score -= 20
            elif size > max_size:
                issues.append(f"Logo too large: {size:.1f}% (maximum: {max_size}%)")
                score -= 10
        except ValueError:
            issues.append("Could not determine logo size")
            score -= 10

        # Placement check
        placement = logo.get('position', '')
        allowed_positions = config.get('logo_allowed_positions', [])
        if placement and allowed_positions and placement not in allowed_positions:
            issues.append(f"Logo placement not optimal: {placement}")
            score -= 10

        # Quality check
        if logo.get('sharpness') == 'blurry':
            issues.append("Logo appears blurry")
            score -= 20

        # Issues check
        logo_issues = logo.get('issues', [])
        if logo_issues:
            for issue in logo_issues:
                issues.append(f"Logo: {issue}")
            score -= 10

    # Color check
    colors = analysis.get('colors', {})
    dominant_colors = colors.get('dominant', [])

    if dominant_colors:
        brand_colors = config.get('brand_colors', [])
        if brand_colors:
            # Check if dominant colors match brand palette
            matches = 0
            for color in dominant_colors:
                for brand_color in brand_colors:
                    if brand_color.lower() in color.lower():
                        matches += 1
                        break

            if matches == 0:
                issues.append("No brand colors detected in image")
                score -= 20
            elif matches < len(dominant_colors) / 2:
                issues.append("Brand colors underrepresented")
                score -= 10

    # Color quality
    if colors.get('quality') == 'washed_out':
        issues.append("Colors appear washed out")
        score -= 10
    elif colors.get('quality') == 'oversaturated':
        issues.append("Colors appear oversaturated")
        score -= 5

    # Style check
    composition = analysis.get('composition', {})

    if composition.get('professional_look') == 'amateur':
        issues.append("Image looks amateur, not professional")
        score -= 20

    if composition.get('clutter'):
        issues.append("Image appears cluttered or messy")
        score -= 15

    if composition.get('distracting_elements'):
        issues.append("Distracting elements present")
        score -= 10

    return {
        'passed': score >= 70,
        'score': max(0.0, score),
        'issues': issues,
        'recommendations': _get_brand_recommendations(issues)
    }


def check_safety(analysis: Dict, config: Dict) -> Dict:
    """
    Check content safety.

    Checks for:
    - Weapons, violence
    - Sexual content, nudity
    - Hate symbols, offensive content
    - Religious imagery (region-specific)
    - Children (special handling)
    - Fraud indicators
    """
    issues = []
    score = 100.0
    critical = False

    safety = analysis.get('content_safety', {})

    # Critical issues (auto-reject)
    if safety.get('weapons'):
        issues.append("Weapons or violent imagery detected")
        score -= 50
        critical = True

    if safety.get('violence'):
        issues.append("Violent content detected")
        score -= 50
        critical = True

    sexual_content = safety.get('sexual_content')
    if sexual_content == 'yes':
        issues.append("Sexual content or nudity detected")
        score -= 50
        critical = True

    if safety.get('hate_symbols'):
        issues.append("Hate symbols or offensive content detected")
        score -= 50
        critical = True

    # Region-specific checks
    religious_symbols = safety.get('religious_symbols')
    if religious_symbols:
        if config.get('restrict_religious_imagery', False):
            issues.append("Religious imagery not allowed in this region")
            score -= 30
        elif config.get('religious_sensitivity', 'low') == 'high':
            issues.append("Religious imagery requires careful review")
            score -= 10

    # Children (always flag)
    if safety.get('children'):
        issues.append("Children depicted - requires special compliance review")
        score -= 20

    # Fraud indicators
    if safety.get('misleading_claims'):
        issues.append("Potentially misleading claims detected")
        score -= 25

    if safety.get('fake_urgency'):
        issues.append("Fake urgency or scarcity tactics detected")
        score -= 15

    # Regional/cultural issues
    cultural_issues = safety.get('cultural_issues', [])
    if cultural_issues:
        for issue in cultural_issues:
            issues.append(f"Cultural sensitivity: {issue}")
            score -= 10

    return {
        'passed': not critical and score >= 70,
        'score': max(0.0, score),
        'issues': issues,
        'critical_issues': critical,
        'recommendations': _get_safety_recommendations(issues, critical)
    }


def check_quality(image_path: str, analysis: Dict) -> Dict:
    """
    Check technical quality.

    Validates:
    - Resolution
    - Sharpness
    - Compression artifacts
    - Lighting
    - Overall quality
    """
    issues = []
    score = 100.0

    # Resolution check
    try:
        with Image.open(image_path) as img:
            width, height = img.size

        min_resolution = 1080
        if width < min_resolution or height < min_resolution:
            issues.append(
                f"Resolution too low: {width}x{height} "
                f"(minimum: {min_resolution}x{min_resolution})"
            )
            score -= 30
        elif width < 1920 or height < 1920:
            # Below optimal but acceptable
            score -= 10

    except Exception as e:
        issues.append(f"Could not verify resolution: {e}")
        score -= 20

    # Sharpness check
    quality = analysis.get('technical_quality', {})
    sharpness = quality.get('sharpness', 'good')

    if sharpness == 'blurry':
        issues.append("Image appears blurry or out of focus")
        score -= 30
    elif sharpness == 'soft':
        issues.append("Image could be sharper")
        score -= 10

    # Artifacts
    if quality.get('artifacts'):
        issues.append("Compression artifacts or noise detected")
        score -= 15

    # Lighting
    lighting = quality.get('lighting_quality', 'good')

    if lighting == 'poor':
        issues.append("Poor lighting quality")
        score -= 20
    elif lighting == 'fair':
        issues.append("Lighting could be improved")
        score -= 10

    # Overall quality
    overall_quality = quality.get('overall_quality', 'good')

    if overall_quality == 'poor':
        issues.append("Overall poor image quality")
        score -= 20
    elif overall_quality == 'fair':
        issues.append("Overall quality could be improved")
        score -= 10

    # Text readability (if text present)
    text = analysis.get('text_overlays', {})
    if text.get('hard_to_read'):
        issues.append("Text overlays are hard to read")
        score -= 15

    text_amount = text.get('amount', 'minimal')
    if text_amount == 'heavy':
        issues.append("Too much text - simplify message")
        score -= 10

    return {
        'passed': score >= 70,
        'score': max(0.0, score),
        'issues': issues,
        'recommendations': _get_quality_recommendations(issues)
    }


def _get_brand_recommendations(issues: List[str]) -> List[str]:
    """Generate brand compliance recommendations."""
    recommendations = []

    for issue in issues:
        issue_lower = issue.lower()

        if 'logo' in issue_lower and 'not detected' in issue_lower:
            recommendations.append("Add brand logo to the creative")
            continue

        if 'logo' in issue_lower and 'small' in issue_lower:
            recommendations.append("Increase logo size to 5-10% of image width")
            continue

        if 'logo' in issue_lower and 'large' in issue_lower:
            recommendations.append("Reduce logo size to under 10% of image width")
            continue

        if 'logo' in issue_lower and 'blurry' in issue_lower:
            recommendations.append("Use sharper, higher-quality logo")
            continue

        if 'color' in issue_lower:
            if 'not detected' in issue_lower or 'underrepresented' in issue_lower:
                recommendations.append("Use brand colors more prominently")
            elif 'washed out' in issue_lower:
                recommendations.append("Increase color saturation for vibrancy")
            elif 'oversaturated' in issue_lower:
                recommendations.append("Reduce color saturation for natural appearance")

        if 'amateur' in issue_lower:
            recommendations.append("Improve composition and professional quality")

        if 'clutter' in issue_lower or 'messy' in issue_lower:
            recommendations.append("Simplify design - remove distracting elements")

        if 'distracting' in issue_lower:
            recommendations.append("Remove or minimize distracting elements")

    return recommendations


def _get_safety_recommendations(issues: List[str], critical: bool) -> List[str]:
    """Generate safety recommendations."""
    recommendations = []

    for issue in issues:
        issue_lower = issue.lower()

        if 'weapon' in issue_lower:
            recommendations.append("Remove all weapons from the image")

        elif 'violence' in issue_lower:
            recommendations.append("Remove all violent or aggressive imagery")

        elif 'sexual' in issue_lower or 'nudity' in issue_lower:
            recommendations.append("Remove all sexual content and nudity")

        elif 'hate' in issue_lower or 'offensive' in issue_lower:
            recommendations.append("Remove hate symbols and offensive content immediately")

        elif 'religious' in issue_lower:
            recommendations.append("Remove religious imagery for this region")

        elif 'child' in issue_lower:
            recommendations.append("Review children depiction for compliance requirements")

        elif 'misleading' in issue_lower:
            recommendations.append("Remove or substantiate all claims")

        elif 'urgency' in issue_lower or 'scarcity' in issue_lower:
            recommendations.append("Remove fake urgency tactics")

        elif 'cultural' in issue_lower:
            recommendations.append("Review for cultural sensitivity")

    return recommendations


def _get_quality_recommendations(issues: List[str]) -> List[str]:
    """Generate quality recommendations."""
    recommendations = []

    for issue in issues:
        issue_lower = issue.lower()

        if 'resolution' in issue_lower:
            recommendations.append("Use higher resolution image (minimum 1080x1080)")

        elif 'blurry' in issue_lower or 'soft' in issue_lower:
            recommendations.append("Use sharper, in-focus image or improve sharpening")

        elif 'artifacts' in issue_lower:
            recommendations.append("Use higher quality export with less compression")

        elif 'lighting' in issue_lower:
            recommendations.append("Improve lighting quality")

        elif 'quality' in issue_lower:
            recommendations.append("Improve overall image quality")

        elif 'text' in issue_lower:
            if 'hard to read' in issue_lower:
                recommendations.append("Improve text contrast and size for readability")
            elif 'too much' in issue_lower:
                recommendations.append("Simplify message - reduce text content")

    return recommendations
```

### 3. GPT-4 Vision Prompts

```python
# src/meta/ad/reviewer/prompts.py

"""
GPT-4 Vision prompts for image analysis.
"""

ANALYSIS_PROMPT = """
Analyze this ad creative image and provide a detailed assessment in JSON format.

Respond ONLY with valid JSON using this exact structure:

{
  "logo": {
    "present": "yes|no",
    "position": "top_left|top_right|center|bottom_left|bottom_right",
    "size_percentage": "estimated size as % of image width (e.g., 5%)",
    "sharpness": "sharp|blurry",
    "issues": ["any specific issues with the logo"]
  },

  "colors": {
    "dominant": ["list 3-5 dominant colors as hex codes or names"],
    "quality": "vibrant|normal|washed_out|oversaturated",
    "variety": "monochromatic|limited|diverse"
  },

  "text_overlays": {
    "present": "yes|no",
    "amount": "minimal|moderate|heavy",
    "hard_to_read": true|false,
    "text_list": ["all text visible in the image"],
    "issues": ["any text readability issues"]
  },

  "composition": {
    "focal_point": "description of main focal point",
    "balanced": true|false,
    "professional_look": "professional|amateur",
    "clutter": true|false,
    "distracting_elements": ["list any distracting elements"]
  },

  "content_safety": {
    "weapons": true|false,
    "violence": true|false,
    "sexual_content": "yes|no",
    "hate_symbols": true|false,
    "religious_symbols": true|false,
    "children": true|false,
    "misleading_claims": true|false,
    "fake_urgency": true|false,
    "cultural_issues": ["any cultural or regional sensitivity issues"]
  },

  "technical_quality": {
    "sharpness": "sharp|good|soft|blurry",
    "artifacts": true|false,
    "lighting_quality": "excellent|good|fair|poor",
    "overall_quality": "excellent|good|fair|poor"
  },

  "overall_assessment": {
    "strengths": ["3-5 key strengths"],
    "weaknesses": ["3-5 key weaknesses or areas for improvement"],
    "single_sentence_summary": "one sentence summary of the creative"
  }
}

Be thorough and specific. Detect all issues, even minor ones.
"""
```

### 4. Data Models

```python
# src/meta/ad/reviewer/models.py

"""
Data models for review results.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import json


@dataclass
class ReviewResult:
    """Result of ad creative review."""
    image_path: str
    brand: str
    region: str
    score: float
    approved: bool
    brand_compliance: Dict
    safety: Dict
    quality: Dict
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'image_path': self.image_path,
            'brand': self.brand,
            'region': self.region,
            'score': round(self.score, 1),
            'approved': self.approved,
            'brand_compliance': self.brand_compliance,
            'safety': self.safety,
            'quality': self.quality,
            'recommendations': self.recommendations
        }

    def save(self, path: str):
        """Save result to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @property
    def critical_issues(self) -> List[str]:
        """Get critical safety issues."""
        return self.safety.get('issues', []) if self.safety.get('critical_issues') else []

    @property
    def brand_issues(self) -> List[str]:
        """Get brand compliance issues."""
        return self.brand_compliance.get('issues', [])

    @property
    def quality_issues(self) -> List[str]:
        """Get quality issues."""
        return self.quality.get('issues', [])

    def __str__(self) -> str:
        """String representation."""
        status = "✅ APPROVED" if self.approved else "❌ REJECTED"
        return (
            f"{status} | Score: {self.score:.0f}/100 | "
            f"Brand: {self.brand} | Region: {self.region}"
        )

    def summary(self) -> str:
        """Generate detailed summary."""
        lines = [
            f"# Ad Review Report",
            f"",
            f"**Image:** {self.image_path}",
            f"**Brand:** {self.brand}",
            f"**Region:** {self.region}",
            f"**Score:** {self.score:.0f}/100",
            f"**Status:** { '✅ APPROVED' if self.approved else '❌ REJECTED' }",
            f"",
            f"## Component Scores",
            f"",
            f"### Brand Compliance: {self.brand_compliance['score']:.0f}/100",
        ]

        if self.brand_issues:
            lines.append(f"**Issues:**")
            for issue in self.brand_issues:
                lines.append(f"  - {issue}")
        else:
            lines.append(f"✅ No brand issues")

        lines.extend([
            f"",
            f"### Safety: {self.safety['score']:.0f}/100",
        ])

        if self.safety.get('issues'):
            lines.append(f"**Issues:**")
            for issue in self.safety['issues']:
                lines.append(f"  - {issue}")
        else:
            lines.append(f"✅ No safety issues")

        lines.extend([
            f"",
            f"### Quality: {self.quality['score']:.0f}/100",
        ])

        if self.quality_issues:
            lines.append(f"**Issues:**")
            for issue in self.quality_issues:
                lines.append(f"  - {issue}")
        else:
            lines.append(f"✅ No quality issues")

        if self.recommendations:
            lines.extend([
                f"",
                f"## Recommendations",
                f"",
            ])
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")

        return "\n".join(lines)
```

### 5. Package Exports

```python
# src/meta/ad/reviewer/__init__.py

"""
Ad Reviewer Package.

Simple, focused ad creative evaluation.
"""

from .reviewer import AdReviewer, ReviewError
from .models import ReviewResult

__all__ = ['AdReviewer', 'ReviewResult', 'ReviewError']
```

---

## Configuration Files

### Brand Configurations

```yaml
# config/ad/reviewer/brands/moprobo.yaml

brand_colors:
  - "#FF0000"  # Moprobo Red
  - "#000000"  # Black
  - "#FFFFFF"  # White
  - "#333333"  # Dark Gray

logo_min_size: 5.0    # % of image width
logo_max_size: 10.0
logo_allowed_positions:
  - top_right
  - top_left

style:
  preferred_aesthetic: "minimalist technical"
  avoid_clutter: true
  professional_quality: true
```

```yaml
# config/ad/reviewer/brands/ecoflow.yaml

brand_colors:
  - "#00A651"  # Eco Green
  - "#2D5A27"  # Leaf Dark Green
  - "#87CEEB"  # Sky Blue
  - "#FFFFFF"  # White

logo_min_size: 4.0
logo_max_size: 8.0
logo_allowed_positions:
  - top_right
  - top_left
  - centered_top

style:
  preferred_aesthetic: "natural outdoor"
  avoid_clutter: false
  professional_quality: true
```

### Regional Configurations

```yaml
# config/ad/reviewer/regions/us.yaml

restrict_religious_imagery: false
religious_sensitivity: low
protect_children: true
allowed_content:
  - alcohol_in_moderation
  - pharmaceutical_with_disclaimer
  - competitive_positioning
```

```yaml
# config/ad/reviewer/regions/sa.yaml

restrict_religious_imagery: true
religious_sensitivity: high
protect_children: true
allowed_content:
  - modest_clothing_only
  - no_alcohol
  - no_religious_symbols
  - gender_appropriate_interactions

cultural_notes:
  - "Ramadan: no food/drink imagery during daylight"
  - "Modesty: no revealing clothing"
  - "Respect: avoid disrespecting Islamic values"
```

```yaml
# config/ad/reviewer/regions/jp.yaml

restrict_religious_imagery: false
religious_sensitivity: low
protect_children: true
allowed_content:
  - minimal_text_preferred
  - subtle_promotions

cultural_notes:
  - "Indirect communication preferred"
  - "Harmony and balance important"
  - "Minimalism valued"
```

---

## Usage Examples

### Basic Usage

```python
from src.meta.ad.reviewer import AdReviewer

# Initialize
reviewer = AdReviewer(brand="moprobo", region="US")

# Review single creative
result = reviewer.review("creative.jpg")

# Check result
print(result)
# ✅ APPROVED | Score: 82/100 | Brand: moprobo | Region: US

if result.approved:
    print(f"Great! Score: {result.score}")
else:
    print(f"Issues found:")
    for rec in result.recommendations:
        print(f"  - {rec}")

# Save report
result.save("review_report.json")

# Get detailed summary
print(result.summary())
```

### Batch Review

```python
# Review multiple creatives
results = reviewer.review_batch([
    "creative1.jpg",
    "creative2.jpg",
    "creative3.jpg"
])

# Summary
approved = sum(1 for r in results.values() if r and r.approved)
total = len(results)
print(f"Approved: {approved}/{total}")

# Show issues for rejected creatives
for path, result in results.items():
    if result and not result.approved:
        print(f"\n{path}:")
        for rec in result.recommendations:
            print(f"  - {rec}")
```

### Different Regions

```python
# Review for Saudi Arabia (stricter)
reviewer_sa = AdReviewer(brand="moprobo", region="SA")
result_sa = reviewer_sa.review("creative.jpg")

# Same creative might have different results
print(f"US: {result.approved}, SA: {result_sa.approved}")
```

---

## Testing

```python
# tests/test_reviewer.py

import pytest
from src.meta.ad.reviewer import AdReviewer

def test_basic_review():
    """Test basic review functionality."""
    reviewer = AdReviewer(brand="moprobo", region="US")
    result = reviewer.review("tests/fixtures/good_creative.jpg")

    assert result.score > 0
    assert isinstance(result.approved, bool)
    assert isinstance(result.recommendations, list)

def test_missing_logo():
    """Test detection of missing logo."""
    reviewer = AdReviewer(brand="moprobo", region="US")
    result = reviewer.review("tests/fixtures/no_logo.jpg")

    assert not result.approved
    assert any('logo' in issue.lower() for issue in result.brand_issues)

def test_low_resolution():
    """Test detection of low resolution."""
    reviewer = AdReviewer(brand="moprobo", region="US")
    result = reviewer.review("tests/fixtures/low_res.jpg")

    assert result.quality['score'] < 70
```

---

## Summary

**Files:** 4 Python files
**Lines:** ~600 total
**Dependencies:** openai, pyyaml, pillow
**Complexity:** Low
**Maintainability:** High

**What it does:**
1. ✅ Analyze creative with GPT-4 Vision (1 call)
2. ✅ Check brand compliance
3. ✅ Check safety issues
4. ✅ Check technical quality
5. ✅ Return score + approve/reject
6. ✅ Provide actionable recommendations

**What it doesn't do (removed for simplicity):**
- ❌ Campaign goal alignment
- ❌ Regional pattern matching
- ❌ Performance prediction
- ❌ Miner/generator evaluation
- ❌ Complex pipelines
- ❌ Multiple report formats

**Simple and focused.**
