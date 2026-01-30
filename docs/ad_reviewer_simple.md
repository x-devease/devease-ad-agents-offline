# Ad Reviewer - Simplified Design

## Principle: Keep It Simple

**Core Problem:** Current design has too many moving parts
**Solution:** Single focused reviewer with 3 essential checks

---

## Simplified Architecture

```
src/meta/ad/reviewer/
├── __init__.py
├── reviewer.py          # Main reviewer class (1 file)
├── checks.py            # All check functions (1 file)
├── prompts.py           # GPT-4 Vision prompts (1 file)
└── models.py            # Data models (1 file)

config/ad/reviewer/
├── brands/              # Brand guidelines (YAML)
│   ├── moprobo.yaml
│   └── ecoflow.yaml
└── regions/             # Risk profiles (YAML)
    ├── us.yaml
    └── sa.yaml
```

**Total: 4 Python files + configs**

---

## 1. Main Reviewer (Single Class)

```python
# src/meta/ad/reviewer/reviewer.py

from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
import json
from openai import OpenAI

from .models import ReviewResult
from .checks import (
    check_brand_guidelines,
    check_safety,
    check_technical_quality
)
from .prompts import QUALITY_PROMPT


class AdReviewer:
    """
    Simple ad reviewer.

    Checks 3 things:
    1. Brand compliance (logo, colors, style)
    2. Safety (religion, violence, fraud, etc.)
    3. Quality (resolution, sharpness, composition)

    Usage:
        reviewer = AdReviewer(brand="moprobo", region="US")
        result = reviewer.review("creative.jpg")
        print(result.score, result.approved)
    """

    def __init__(
        self,
        brand: str,
        region: str = "US",
        openai_api_key: Optional[str] = None
    ):
        self.brand = brand
        self.region = region
        self.client = OpenAI(api_key=openai_api_key)

        # Load configs
        self.brand_config = self._load_yaml(f"config/ad/reviewer/brands/{brand}.yaml")
        self.region_config = self._load_yaml(f"config/ad/reviewer/regions/{region}.yaml")

    def review(self, image_path: str) -> ReviewResult:
        """Review a creative image."""

        # Step 1: Analyze with GPT-4 Vision (single call)
        analysis = self._analyze_image(image_path)

        # Step 2: Run checks
        brand_result = check_brand_guidelines(analysis, self.brand_config)
        safety_result = check_safety(analysis, self.region_config)
        quality_result = check_technical_quality(image_path, analysis)

        # Step 3: Calculate score
        score = self._calculate_score(brand_result, safety_result, quality_result)

        # Step 4: Determine approval
        approved = (
            score >= 70 and
            brand_result['passed'] and
            safety_result['passed'] and
            quality_result['passed']
        )

        # Step 5: Generate recommendations
        recommendations = (
            brand_result.get('recommendations', []) +
            safety_result.get('recommendations', []) +
            quality_result.get('recommendations', [])
        )

        return ReviewResult(
            image_path=image_path,
            score=score,
            approved=approved,
            brand_compliance=brand_result,
            safety=safety_result,
            quality=quality_result,
            recommendations=recommendations
        )

    def _analyze_image(self, image_path: str) -> Dict:
        """Single GPT-4 Vision call for all analysis."""

        base64_image = self._encode_image(image_path)

        prompt = """
        Analyze this ad creative image and provide:

        1. **Logo Detection**:
           - Is there a logo? (yes/no)
           - Where is it positioned? (top_left, top_right, center, bottom_left, bottom_right)
           - Approximate size as % of image width
           - Is it sharp or blurry?
           - Any issues with the logo?

        2. **Colors**:
           - List dominant colors (hex codes if possible)
           - Are colors vibrant or muted?
           - Any color issues (washed out, oversaturated)?

        3. **Text Overlays**:
           - List all text visible in the image
           - For each text: position, approximate size, color
           - Any text cut off or hard to read?
           - Overall text amount (minimal, moderate, heavy)

        4. **Composition**:
           - What's the main focal point?
           - Is the image balanced? (yes/no)
           - Does it look professional or amateur?
           - Any clutter or distracting elements?

        5. **Content Safety**:
           - Any religious symbols or imagery?
           - Any weapons or violence?
           - Any sexual content or nudity?
           - Any hate symbols or offensive content?
           - Any children in the image?

        6. **Technical Quality**:
           - Does the image look sharp or blurry?
           - Any compression artifacts or noise?
           - Lighting quality (good, fair, poor)
           - Overall professional quality (excellent, good, fair, poor)

        Return as JSON.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=1500,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    def _calculate_score(self, brand: Dict, safety: Dict, quality: Dict) -> float:
        """Calculate overall score (0-100)."""
        brand_score = brand['score']
        safety_score = safety['score']
        quality_score = quality['score']

        # Weighted average
        return (brand_score * 0.4 + safety_score * 0.3 + quality_score * 0.3)

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        import base64
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _load_yaml(self, path: str) -> Dict:
        """Load YAML config."""
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
```

---

## 2. Check Functions (Single File)

```python
# src/meta/ad/reviewer/checks.py

from typing import Dict, List


def check_brand_guidelines(analysis: Dict, config: Dict) -> Dict:
    """Check brand compliance."""

    issues = []
    score = 100.0

    # 1. Logo check
    logo = analysis.get('logo', {})
    if logo.get('present') != 'yes':
        issues.append("Logo not detected")
        score -= 50
    else:
        # Check size
        size_str = logo.get('size_percentage', '0%')
        size = float(size_str.rstrip('%'))
        min_size = config.get('logo_min_size', 5.0)
        max_size = config.get('logo_max_size', 10.0)

        if size < min_size:
            issues.append(f"Logo too small: {size}% (min: {min_size}%)")
            score -= 20
        elif size > max_size:
            issues.append(f"Logo too large: {size}% (max: {max_size}%)")
            score -= 10

        # Check placement
        placement = logo.get('position', '')
        allowed = config.get('logo_allowed_positions', ['top_right', 'top_left'])
        if placement and placement not in allowed:
            issues.append(f"Logo placement not optimal: {placement}")
            score -= 10

        # Check quality
        if logo.get('sharpness') == 'blurry':
            issues.append("Logo appears blurry")
            score -= 20

    # 2. Color check
    colors = analysis.get('colors', {})
    dominant_colors = colors.get('dominant', [])

    if dominant_colors:
        brand_colors = config.get('brand_colors', [])
        # Simple check: do dominant colors match brand?
        matches = sum(
            1 for c in dominant_colors
            if any(bc.lower() in c.lower() for bc in brand_colors)
        )
        if matches < len(dominant_colors) * 0.5:
            issues.append("Colors don't match brand palette")
            score -= 15

    # 3. Style check
    composition = analysis.get('composition', {})
    if composition.get('professional_look') == 'amateur':
        issues.append("Image looks amateur, not professional")
        score -= 20

    if composition.get('clutter'):
        issues.append("Image appears cluttered")
        score -= 10

    return {
        'passed': score >= 70,
        'score': max(0, score),
        'issues': issues,
        'recommendations': _get_brand_recommendations(issues)
    }


def check_safety(analysis: Dict, config: Dict) -> Dict:
    """Check content safety."""

    issues = []
    score = 100.0
    critical = False

    safety = analysis.get('content_safety', {})

    # Critical issues (auto-reject)
    if safety.get('weapons'):
        issues.append("Weapons detected")
        score -= 50
        critical = True

    if safety.get('violence'):
        issues.append("Violence detected")
        score -= 50
        critical = True

    if safety.get('sexual_content') == 'yes':
        issues.append("Sexual content detected")
        score -= 50
        critical = True

    if safety.get('hate_symbols'):
        issues.append("Hate symbols detected")
        score -= 50
        critical = True

    # Regional/cultural issues
    religious = safety.get('religious_symbols')
    if religious and config.get('restrict_religious_imagery', False):
        issues.append("Religious imagery not allowed in this region")
        score -= 30

    # Children (needs special care)
    if safety.get('children'):
        if config.get('protect_children', True):
            issues.append("Children in image - requires special review")
            score -= 20

    return {
        'passed': not critical and score >= 70,
        'score': max(0, score),
        'issues': issues,
        'critical_issues': critical,
        'recommendations': _get_safety_recommendations(issues, critical)
    }


def check_technical_quality(image_path: str, analysis: Dict) -> Dict:
    """Check technical quality."""

    issues = []
    score = 100.0

    # 1. Resolution check
    from PIL import Image
    with Image.open(image_path) as img:
        width, height = img.size

    if width < 1080 or height < 1080:
        issues.append(f"Resolution too low: {width}x{height} (min: 1080x1080)")
        score -= 30

    # 2. Sharpness check
    quality = analysis.get('technical_quality', {})
    if quality.get('sharpness') == 'blurry':
        issues.append("Image appears blurry")
        score -= 30

    # 3. Artifacts
    if quality.get('artifacts'):
        issues.append("Compression artifacts detected")
        score -= 15

    # 4. Lighting
    lighting = quality.get('lighting_quality', 'good')
    if lighting == 'poor':
        issues.append("Poor lighting quality")
        score -= 20
    elif lighting == 'fair':
        score -= 10

    # 5. Overall quality
    overall = quality.get('overall_quality', 'good')
    if overall == 'poor':
        issues.append("Overall poor quality")
        score -= 20
    elif overall == 'fair':
        score -= 10

    return {
        'passed': score >= 70,
        'score': max(0, score),
        'issues': issues,
        'recommendations': _get_quality_recommendations(issues)
    }


def _get_brand_recommendations(issues: List[str]) -> List[str]:
    """Generate brand compliance recommendations."""
    recommendations = []

    for issue in issues:
        if 'logo' in issue.lower() and 'not detected' in issue.lower():
            recommendations.append("Add brand logo to the creative")
        elif 'logo' in issue.lower() and 'small' in issue.lower():
            recommendations.append("Increase logo size to 5-10% of image width")
        elif 'logo' in issue.lower() and 'large' in issue.lower():
            recommendations.append("Reduce logo size to under 10% of image width")
        elif 'colors' in issue.lower():
            recommendations.append("Use brand color palette more prominently")
        elif 'amateur' in issue.lower():
            recommendations.append("Improve composition and professional quality")
        elif 'clutter' in issue.lower():
            recommendations.append("Simplify design and remove distracting elements")

    return recommendations


def _get_safety_recommendations(issues: List[str], critical: bool) -> List[str]:
    """Generate safety recommendations."""
    recommendations = []

    for issue in issues:
        if 'weapon' in issue.lower():
            recommendations.append("Remove all weapons from the image")
        elif 'violence' in issue.lower():
            recommendations.append("Remove violent imagery")
        elif 'sexual' in issue.lower():
            recommendations.append("Remove all sexual content")
        elif 'hate' in issue.lower():
            recommendations.append("Remove hate symbols immediately")
        elif 'religious' in issue.lower():
            recommendations.append("Remove religious imagery for this region")
        elif 'children' in issue.lower():
            recommendations.append("Review children depiction for compliance")

    if critical:
        recommendations.insert(0, "CRITICAL: Address safety issues before use")

    return recommendations


def _get_quality_recommendations(issues: List[str]) -> List[str]:
    """Generate quality recommendations."""
    recommendations = []

    for issue in issues:
        if 'resolution' in issue.lower():
            recommendations.append("Use higher resolution image (minimum 1080x1080)")
        elif 'blurry' in issue.lower():
            recommendations.append("Use sharper, in-focus image")
        elif 'artifacts' in issue.lower():
            recommendations.append("Use higher quality export with less compression")
        elif 'lighting' in issue.lower():
            recommendations.append("Improve lighting quality")
        elif 'quality' in issue.lower():
            recommendations.append("Overall quality needs improvement")

    return recommendations
```

---

## 3. Data Models (Single File)

```python
# src/meta/ad/reviewer/models.py

from dataclasses import dataclass, field
from typing import Dict, List
import json


@dataclass
class ReviewResult:
    """Result of ad review."""
    image_path: str
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
            'score': self.score,
            'approved': self.approved,
            'brand_compliance': self.brand_compliance,
            'safety': self.safety,
            'quality': self.quality,
            'recommendations': self.recommendations
        }

    def save(self, path: str):
        """Save to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def __str__(self) -> str:
        """String summary."""
        status = "✅ APPROVED" if self.approved else "❌ REJECTED"
        return f"{status} | Score: {self.score:.0f}/100"
```

---

## 4. Configuration Files (Simple YAML)

```yaml
# config/ad/reviewer/brands/moprobo.yaml

brand_colors:
  - "#FF0000"  # Red
  - "#000000"  # Black
  - "#FFFFFF"  # White

logo_min_size: 5.0      # % of image width
logo_max_size: 10.0
logo_allowed_positions:
  - top_right
  - top_left

style:
  preferred: "minimalist technical"
  avoid_clutter: true
```

```yaml
# config/ad/reviewer/regions/us.yaml

restrict_religious_imagery: false
protect_children: true
allowed_content:
  - alcohol_in_moderation
  - pharmaceutical_with_disclaimer
```

```yaml
# config/ad/reviewer/regions/sa.yaml

restrict_religious_imagery: true
protect_children: true
allowed_content:
  - modest_clothing_only
  - no_alcohol
  - no_religious_symbols
```

---

## 5. Usage

```python
# Single file usage
from src.meta.ad.reviewer import AdReviewer

# Initialize
reviewer = AdReviewer(brand="moprobo", region="US")

# Review
result = reviewer.review("creative.jpg")

# Check result
if result.approved:
    print(f"✅ Approved! Score: {result.score}")
else:
    print(f"❌ Rejected! Score: {result.score}")
    print("\nIssues:")
    for issue in result.brand_compliance['issues']:
        print(f"  - Brand: {issue}")
    for issue in result.safety['issues']:
        print(f"  - Safety: {issue}")
    for issue in result.quality['issues']:
        print(f"  - Quality: {issue}")

    print("\nRecommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")

# Save report
result.save("review_report.json")
```

---

## Comparison: Before vs After

| Aspect | Complex Design | Simple Design |
|---|---|---|
| **Python Files** | 15+ files | 4 files |
| **Classes** | 10+ classes | 1 main class |
| **Check Types** | 10 dimensions | 3 checks |
| **Config Files** | Multiple nested configs | 2 simple YAMLs |
| **GPT-4 Calls** | Multiple calls | 1 single call |
| **Abstraction** | Deep (base classes, factories) | Flat (functions) |
| **Setup Time** | Hours | Minutes |
| **Mental Load** | High | Low |

---

## What Was Removed

❌ **Removed:**
- Campaign goal alignment (6 different goal types)
- Regional pattern matching (historical performance data)
- Miner/Generator evaluation
- Performance prediction (ML models)
- Multiple analyzer classes
- Base checker classes
- Evaluator classes
- Predictor classes
- Pipeline orchestrator
- Batch processor
- Multiple report formats
- Complex configuration system

✅ **Kept:**
- Brand compliance (logo, colors, style)
- Safety checks (religion, violence, fraud, etc.)
- Technical quality (resolution, sharpness, artifacts)

---

## What You Get

**Minimum Viable Reviewer:**
1. ✅ Checks brand compliance
2. ✅ Catches safety issues
3. ✅ Ensures technical quality
4. ✅ Single GPT-4 Vision call
5. ✅ Simple score (0-100)
6. ✅ Clear approve/reject
7. ✅ Actionable recommendations

**Total: ~400 lines of code** (vs 3000+ in complex version)

---

## When to Add Complexity

Only add complexity when you have a clear need:

1. **Need historical performance prediction?** → Add regional matcher
2. **Need to evaluate miner/generator?** → Add evaluators
3. **Need batch processing?** → Add simple loop (no need for complex orchestrator)
4. **Need advanced features?** → Add one at a time, as needed

**Start simple. Add complexity only when proven necessary.**
