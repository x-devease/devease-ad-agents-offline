# Ad Creative Quality Checking - Complete Specification

## Overview

Quality checking ensures creatives meet technical and aesthetic standards before launch. This specification covers:

1. **Technical Quality** - Resolution, sharpness, file integrity, artifacts
2. **Visual Quality** - Composition, balance, aesthetic appeal
3. **Text Quality** - Readability, contrast, font rendering
4. **Brand Quality** - Logo quality, color accuracy, consistency

---

## 1. Technical Quality Checks

### 1.1 Resolution and Dimensions

```python
# src/meta/ad/qa/checkers/technical_quality_checker.py

from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class TechnicalQualityChecker:
    """Checks technical quality of ad creatives."""

    def __init__(self, config: CriteriaConfig):
        self.config = config
        self.min_resolution = config.min_resolution  # (1080, 1080)
        self.max_blur_score = config.max_blur_score  # 0.3
        self.max_file_size_mb = config.max_file_size_mb  # 50.0

    def check_resolution(self, image_path: str) -> Dict[str, any]:
        """
        Check image resolution and dimensions.

        Returns:
            Dict with resolution check results
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                resolution = (width, height)

                # Check if meets minimum resolution
                min_width, min_height = self.min_resolution
                meets_min = width >= min_width and height >= min_height

                # Check aspect ratio
                aspect_ratio = width / height
                is_square = 0.95 <= aspect_ratio <= 1.05
                is_landscape = aspect_ratio > 1.05
                is_portrait = aspect_ratio < 0.95

                # Platform-specific recommendations
                platform_recommendations = self._get_platform_recommendations(
                    width, height
                )

                # Calculate quality score
                if meets_min:
                    # Higher resolution = better quality (up to 4K)
                    megapixels = (width * height) / 1_000_000
                    if megapixels >= 8:  # 4K and above
                        score = 100.0
                    elif megapixels >= 2:  # 1080p
                        score = 95.0
                    else:
                        score = 80.0
                else:
                    score = 0.0

                return {
                    'compliant': meets_min,
                    'score': score,
                    'detected': {
                        'width': width,
                        'height': height,
                        'aspect_ratio': round(aspect_ratio, 2),
                        'megapixels': round(megapixels, 2)
                    },
                    'required': {
                        'min_width': min_width,
                        'min_height': min_height,
                        'min_aspect_ratio': 'varies_by_platform'
                    },
                    'aspect_ratio_type': 'square' if is_square else 'landscape' if is_landscape else 'portrait',
                    'platform_recommendations': platform_recommendations,
                    'issue': None if meets_min else f"Resolution {width}x{height} below minimum {min_width}x{min_height}",
                    'recommendation': None if meets_min else f"Increase resolution to at least {min_width}x{min_height}"
                }

        except Exception as e:
            logger.error(f"Failed to check resolution: {e}")
            return {
                'compliant': False,
                'score': 0.0,
                'error': str(e)
            }

    def _get_platform_recommendations(
        self,
        width: int,
        height: int
    ) -> List[Dict[str, str]]:
        """Get platform-specific recommendations."""
        recommendations = []

        # Meta/Facebook recommendations
        if width == height:
            recommendations.append({
                'platform': 'meta',
                'recommendation': 'square',
                'optimal': '1080x1080',
                'status': 'optimal' if width >= 1080 else 'acceptable'
            })
        elif width / height == 1.91:  # 1200x628
            recommendations.append({
                'platform': 'meta',
                'recommendation': 'landscape',
                'optimal': '1200x628',
                'status': 'optimal'
            })
        elif width / height == 4 / 5:  # 1080x1350
            recommendations.append({
                'platform': 'meta',
                'recommendation': 'portrait',
                'optimal': '1080x1350',
                'status': 'optimal'
            })
        else:
            recommendations.append({
                'platform': 'meta',
                'recommendation': 'Use standard aspect ratios',
                'optimal': '1:1, 1.91:1, or 4:5',
                'status': 'suboptimal'
            })

        # Instagram specific
        if width == height and width >= 1080:
            recommendations.append({
                'platform': 'instagram',
                'recommendation': 'square',
                'optimal': '1080x1080',
                'status': 'optimal'
            })

        # Stories/Reels
        if width / height == 9 / 16:
            recommendations.append({
                'platform': 'instagram_stories',
                'recommendation': 'vertical',
                'optimal': '1080x1920',
                'status': 'optimal'
            })

        return recommendations

    def check_file_size(self, image_path: str) -> Dict[str, any]:
        """
        Check file size and format.

        Returns:
            Dict with file size check results
        """
        path = Path(image_path)
        file_size_bytes = path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        # Check format
        with Image.open(image_path) as img:
            format = img.format  # JPEG, PNG, etc.
            mode = img.mode  # RGB, RGBA, etc.

        # Check if file size acceptable
        size_ok = file_size_mb <= self.max_file_size_mb

        # Quality recommendations
        quality_score = 100.0

        if file_size_mb < 0.01:  # Less than 10KB - likely very low quality
            quality_score -= 30.0
        elif file_size_mb < 0.05:  # Less than 50KB - potentially low quality
            quality_score -= 10.0
        elif file_size_mb > self.max_file_size_mb:
            quality_score = 0.0

        # Format recommendations
        format_recommendations = []
        if format == 'PNG':
            # PNG for graphics with transparency
            if mode == 'RGBA':
                format_recommendations.append('PNG appropriate for transparent background')
            else:
                format_recommendations.append('Consider JPEG for photos (smaller file size)')
        elif format == 'JPEG':
            if file_size_mb > 1.0:
                # Could be compressed more
                quality_score -= 5.0
                format_recommendations.append('Consider higher compression for faster loading')

        return {
            'compliant': size_ok,
            'score': quality_score,
            'detected': {
                'size_bytes': file_size_bytes,
                'size_mb': round(file_size_mb, 2),
                'format': format,
                'mode': mode
            },
            'required': {
                'max_size_mb': self.max_file_size_mb,
                'accepted_formats': ['JPEG', 'PNG']
            },
            'issue': None if size_ok else f"File size {file_size_mb:.2f}MB exceeds maximum {self.max_file_size_mb}MB",
            'recommendation': None if size_ok else f"Compress image to under {self.max_file_size_mb}MB",
            'format_recommendations': format_recommendations
        }
```

### 1.2 Sharpness and Blur Detection

```python
    def check_sharpness(self, image_path: str) -> Dict[str, any]:
        """
        Check image sharpness using Laplacian variance.

        High variance = sharp image
        Low variance = blurry image

        Returns:
            Dict with sharpness check results
        """
        import cv2
        import numpy as np

        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {
                    'compliant': False,
                    'score': 0.0,
                    'error': 'Could not load image'
                }

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Calculate Laplacian variance (measure of blur)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Thresholds for blur detection
            # These values are empirical and may need calibration
            BLURRY_THRESHOLD = 100.0
            OKAY_THRESHOLD = 500.0

            if laplacian_var < BLURRY_THRESHOLD:
                blur_score = 1.0  # Very blurry
                rating = 'very_blurry'
                compliant = False
            elif laplacian_var < OKAY_THRESHOLD:
                blur_score = laplacian_var / OKAY_THRESHOLD  # 0-1 scale
                rating = 'acceptable' if blur_score > 0.5 else 'blurry'
                compliant = blur_score > 0.3
            else:
                blur_score = 1.0  # Sharp
                rating = 'sharp'
                compliant = True

            # Map 0-1 blur score to quality score
            quality_score = blur_score * 100.0

            return {
                'compliant': compliant,
                'score': quality_score,
                'detected': {
                    'laplacian_variance': round(laplacian_var, 2),
                    'blur_score': round(blur_score, 3),
                    'rating': rating
                },
                'thresholds': {
                    'blurry_threshold': BLURRY_THRESHOLD,
                    'okay_threshold': OKAY_THRESHOLD,
                    'max_blur_score': self.max_blur_score
                },
                'issue': None if compliant else f"Image appears blurry (variance: {laplacian_var:.2f})",
                'recommendation': None if compliant else "Use higher quality source image or check focus"
            }

        except Exception as e:
            logger.error(f"Failed to check sharpness: {e}")
            return {
                'compliant': False,
                'score': 0.0,
                'error': str(e)
            }

    def check_edge_quality(self, image_path: str) -> Dict[str, any]:
        """
        Check edge quality and clarity.

        Returns:
            Dict with edge quality check results
        """
        import cv2
        import numpy as np

        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Canny edge detection
            edges = cv2.Canny(gray, 100, 200)

            # Count edge pixels
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            edge_ratio = edge_pixels / total_pixels

            # Good images typically have 5-15% edge pixels
            # Too few = blurry/flat
            # Too many = noisy
            if edge_ratio < 0.02:
                quality = 'low_edges'
                score = 50.0
                compliant = False
            elif edge_ratio < 0.05:
                quality = 'acceptable_edges'
                score = 80.0
                compliant = True
            elif edge_ratio < 0.15:
                quality = 'good_edges'
                score = 100.0
                compliant = True
            else:
                quality = 'noisy'
                score = 70.0
                compliant = True

            return {
                'compliant': compliant,
                'score': score,
                'detected': {
                    'edge_ratio': round(edge_ratio, 4),
                    'edge_pixels': edge_pixels,
                    'quality': quality
                },
                'issue': None if compliant else f"Low edge count suggests blurry image",
                'recommendation': None if compliant else "Use sharper source image"
            }

        except Exception as e:
            logger.error(f"Failed to check edge quality: {e}")
            return {
                'compliant': False,
                'score': 0.0,
                'error': str(e)
            }
```

### 1.3 Artifact Detection

```python
    def check_artifacts(self, image_path: str, image_analysis: Dict) -> Dict[str, any]:
        """
        Check for compression artifacts and other quality issues.

        Returns:
            Dict with artifact check results
        """
        import cv2
        import numpy as np

        try:
            img = cv2.imread(image_path)

            # Check for JPEG compression artifacts (blockiness)
            # Convert to grayscale and check for 8x8 blocks
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Calculate high-frequency content (compression reduces high freq)
            # Using FFT
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.log(np.abs(fshift) + 1)

            # High frequency energy
            h, w = magnitude_spectrum.shape
            high_freq_region = magnitude_spectrum[h//2-50:h//2+50, w//2-50:w//2+50]
            high_freq_energy = np.mean(high_freq_region)

            # Compression artifacts detection
            artifacts = []
            score = 100.0

            # Check for blocking artifacts
            if high_freq_energy < 2.0:
                artifacts.append('compression_artifacts')
                score -= 20.0

            # Check for noise
            noise_level = self._estimate_noise(gray)
            if noise_level > 20:
                artifacts.append('excessive_noise')
                score -= 15.0

            # Check for posterization (color banding)
            if self._has_posterization(img):
                artifacts.append('posterization')
                score -= 10.0

            # Check from GPT-4 Vision analysis
            vision_artifacts = image_analysis.get('artifacts', [])
            artifacts.extend(vision_artifacts)

            # Unique artifacts only
            artifacts = list(set(artifacts))

            compliant = len(artifacts) == 0 or score >= 70.0

            return {
                'compliant': compliant,
                'score': max(0.0, score),
                'detected': {
                    'artifacts': artifacts,
                    'high_freq_energy': round(high_freq_energy, 2),
                    'noise_level': round(noise_level, 2) if noise_level else None
                },
                'issue': None if compliant else f"Artifacts detected: {', '.join(artifacts)}",
                'recommendation': None if compliant else "Use higher quality export or different compression settings"
            }

        except Exception as e:
            logger.error(f"Failed to check artifacts: {e}")
            return {
                'compliant': False,
                'score': 0.0,
                'error': str(e)
            }

    def _estimate_noise(self, gray_image: np.ndarray) -> float:
        """Estimate noise level in image."""
        # Use Laplacian to estimate noise
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        noise_std = np.std(laplacian)
        return noise_std

    def _has_posterization(self, image: np.ndarray) -> bool:
        """Check for color posterization (banding)."""
        # Reduce to 64 colors and check compression
        # Simple heuristic: check color distribution
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        total_pixels = image.shape[0] * image.shape[1]
        color_ratio = unique_colors / total_pixels

        # If very few unique colors relative to image size, likely posterized
        return color_ratio < 0.001
```

### 1.4 Color Quality

```python
    def check_color_quality(self, image_path: str, color_analysis: Dict) -> Dict[str, any]:
        """
        Check color quality and vibrancy.

        Returns:
            Dict with color quality check results
        """
        import cv2
        import numpy as np

        try:
            img = cv2.imread(image_path)

            # Convert to different color spaces
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

            # Check saturation (vibrancy)
            saturation = hsv[:, :, 1].mean()
            saturation_norm = saturation / 255.0  # 0-1 scale

            # Check brightness/value
            value = hsv[:, :, 2].mean()
            value_norm = value / 255.0

            # Check overall colorfulness
            # Using standard deviation of A and B channels in LAB
            a_std = lab[:, :, 1].std()
            b_std = lab[:, :, 2].std()
            colorfulness = (a_std + b_std) / 2

            issues = []
            score = 100.0

            # Check for too low saturation (washed out)
            if saturation_norm < 0.2:
                issues.append('low_saturation')
                score -= 15.0

            # Check for too high saturation (oversaturated)
            if saturation_norm > 0.9:
                issues.append('oversaturated')
                score -= 10.0

            # Check for too dark
            if value_norm < 0.2:
                issues.append('too_dark')
                score -= 20.0

            # Check for too bright
            if value_norm > 0.9:
                issues.append('too_bright')
                score -= 10.0

            # Check color variety
            dominant_colors = color_analysis.get('dominant', [])
            if len(dominant_colors) < 3:
                issues.append('low_color_variety')
                score -= 10.0

            compliant = score >= 70.0

            return {
                'compliant': compliant,
                'score': max(0.0, score),
                'detected': {
                    'saturation': round(saturation_norm, 3),
                    'brightness': round(value_norm, 3),
                    'colorfulness': round(colorfulness, 2),
                    'color_count': len(dominant_colors),
                    'issues': issues
                },
                'issue': None if compliant else f"Color quality issues: {', '.join(issues)}",
                'recommendation': self._get_color_recommendation(issues, saturation_norm, value_norm)
            }

        except Exception as e:
            logger.error(f"Failed to check color quality: {e}")
            return {
                'compliant': False,
                'score': 0.0,
                'error': str(e)
            }

    def _get_color_recommendation(self, issues: List, saturation: float, brightness: float) -> str:
        """Get color quality recommendation."""
        if not issues:
            return None

        recommendations = []

        if 'low_saturation' in issues:
            recommendations.append("Increase saturation for more vibrant colors")

        if 'oversaturated' in issues:
            recommendations.append("Reduce saturation for more natural colors")

        if 'too_dark' in issues:
            recommendations.append("Increase brightness for better visibility")

        if 'too_bright' in issues:
            recommendations.append("Reduce brightness to avoid washed-out appearance")

        if 'low_color_variety' in issues:
            recommendations.append("Add more color variety for visual interest")

        return "; ".join(recommendations)
```

---

## 2. Visual Quality Checks

### 2.1 GPT-4 Vision Prompts for Quality Assessment

```python
# src/meta/ad/qa/analyzers/vision_analyzer.py

class VisionAnalyzer:
    """GPT-4 Vision integration for image analysis."""

    def __init__(self):
        self.client = openai.OpenAI()

    def analyze_quality(self, image_path: str) -> Dict[str, any]:
        """
        Analyze visual quality using GPT-4 Vision.

        Returns:
            Dict with quality assessment
        """
        # Base64 encode image
        base64_image = self._encode_image(image_path)

        # Quality assessment prompt
        quality_prompt = """
        Analyze this image for visual quality issues. Provide a detailed assessment of:

        1. **Composition Quality**: Is the image well-composed? Check for:
           - Rule of thirds alignment
           - Balanced visual weight
           - Clear focal point
           - Appropriate negative space
           - Professional framing

        2. **Lighting Quality**: Assess lighting:
           - Even lighting (no harsh shadows or hotspots)
           - Appropriate brightness
           - Good contrast
           - Professional look
           - Natural vs artificial lighting

        3. **Color Quality**: Evaluate colors:
           - Color harmony and balance
           - Appropriate saturation (not washed out or oversaturated)
           - Consistent color temperature
           - Professional color grading

        4. **Professional Polish**:
           - Overall aesthetic appeal
           - Looks professionally created
           - Attention to detail
           - Commercial/advertising quality

        5. **Specific Issues**:
           - Any amateurish elements
           - Poor cropping or framing
           - Distracting elements
           - Inconsistent style
           - Stock photo appearance

        Return JSON in this exact format:
        {
            "composition_quality": {
                "score": <0-100>,
                "follows_rule_of_thirds": <boolean>,
                "has_balance": <boolean>,
                "clear_focal_point": <boolean>,
                "negative_space_appropriate": <boolean>,
                "professional_framing": <boolean>,
                "issues": ["<list of specific issues>"],
                "strengths": ["<list of specific strengths>"]
            },
            "lighting_quality": {
                "score": <0-100>,
                "even_lighting": <boolean>,
                "appropriate_brightness": <boolean>,
                "good_contrast": <boolean>,
                "professional_look": <boolean>,
                "lighting_type": "<natural|artificial|mixed|studio>",
                "issues": ["<list>"],
                "strengths": ["<list>"]
            },
            "color_quality": {
                "score": <0-100>,
                "color_harmony": <boolean>,
                "appropriate_saturation": <boolean>,
                "consistent_temperature": <boolean>,
                "professional_grading": <boolean>,
                "dominant_mood": "<professional|playful|warm|cool|dramatic>",
                "issues": ["<list>"],
                "strengths": ["<list>"]
            },
            "professional_polish": {
                "score": <0-100>,
                "aesthetic_appeal": <high|medium|low>,
                "looks_professional": <boolean>,
                "attention_to_detail": <high|medium|low>,
                "commercial_quality": <boolean>,
                "amateurish_elements": ["<list>"],
                "distracting_elements": ["<list>"],
                "looks_like_stock": <boolean>
            },
            "overall_quality_score": <0-100>,
            "quality_rating": "<excellent|good|acceptable|poor>",
            "primary_quality_issues": ["<top 3 issues>"],
            "primary_quality_strengths": ["<top 3 strengths>"],
            "improvement_suggestions": ["<specific actionable suggestions>"]
        }
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": quality_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        # Parse response
        import json
        result = json.loads(response.choices[0].message.content)

        return {
            'composition_quality': result['composition_quality'],
            'lighting_quality': result['lighting_quality'],
            'color_quality': result['color_quality'],
            'professional_polish': result['professional_polish'],
            'overall_quality_score': result['overall_quality_score'],
            'quality_rating': result['quality_rating'],
            'primary_issues': result['primary_quality_issues'],
            'primary_strengths': result['primary_quality_strengths'],
            'improvement_suggestions': result['improvement_suggestions']
        }

    def check_aesthetic_composition(self, image_path: str) -> Dict[str, any]:
        """
        Detailed composition analysis.

        Returns:
            Dict with composition analysis
        """
        base64_image = self._encode_image(image_path)

        composition_prompt = """
        Analyze the composition of this advertising image in detail:

        1. **Visual Hierarchy**:
           - What is the primary focal point?
           - Is the hierarchy clear and intentional?
           - Does eye flow make sense?

        2. **Balance**:
           - Is the image balanced?
           - Type of balance (symmetrical, asymmetrical, radial)?
           - Visual weight distribution

        3. **Rule of Thirds**:
           - Does it follow rule of thirds?
           - Are key elements on power points?
           - Centered vs. off-center composition

        4. **Negative Space**:
           - Is there appropriate breathing room?
           - Is negative space used effectively?
           - Is it too cluttered or too sparse?

        5. **Depth and Layers**:
           - Does it have visual depth?
           - Clear foreground, middle ground, background?
           - Layering creates interest?

        6. **Alignment and Grid**:
           - Elements aligned properly?
           - Consistent spacing and margins?
           - Professional layout?

        7. **Color Harmony**:
           - Do colors work well together?
           - Color scheme (monochromatic, complementary, analogous, triadic)?
           - Emotional response to colors?

        Return JSON:
        {
            "focal_point": {
                "identified": <boolean>,
                "description": "<what catches attention first>",
                "location": "<top_left|top_center|top_right|center|bottom_left|bottom_center|bottom_right>",
                "clarity": <clear|somewhat_clear|unclear>
            },
            "visual_hierarchy": {
                "clear": <boolean>,
                "layers": [
                    {"order": 1, "element": "<description>"},
                    {"order": 2, "element": "<description>"},
                    {"order": 3, "element": "<description>"}
                ],
                "eye_flow": "<logical|confusing|neutral>"
            },
            "balance": {
                "type": "<symmetrical|asymmetrical|radical|unbalanced>",
                "score": <0-100>,
                "left_right_balance": "<left_heavy|balanced|right_heavy>",
                "top_bottom_balance": "<top_heavy|balanced|bottom_heavy>"
            },
            "rule_of_thirds": {
                "followed": <boolean>,
                "key_elements_on_power_points": <int>,
                "centered": <boolean>,
                "description": "<analysis>"
            },
            "negative_space": {
                "appropriate": <boolean>,
                "amount": "<too_little|appropriate|too_much>",
                "used_effectively": <boolean>,
                "percentage_estimate": <0-100>
            },
            "depth": {
                "has_depth": <boolean>,
                "layers": <0-3>,
                "foreground": "<description>",
                "middle_ground": "<description>",
                "background": "<description>"
            },
            "alignment": {
                "professional": <boolean>,
                "consistent_spacing": <boolean>,
                "grid_aligned": <boolean>,
                "issues": ["<list of alignment issues>"]
            },
            "color_harmony": {
                "scheme": "<monochromatic|complementary|analogous|triadic|tetradic|neutral>",
                "harmonious": <boolean>,
                "emotional_impact": "<calm|energetic|professional|playful|dramatic>",
                "clashes": ["<any color clashes>"]
            },
            "overall_composition_score": <0-100>,
            "composition_rating": "<excellent|good|acceptable|needs_improvement>",
            "specific_composition_issues": ["<list>"],
            "composition_improvements": ["<list>"]
        }
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": composition_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        import json
        result = json.loads(response.choices[0].message.content)

        return result

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        import base64

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
```

### 2.2 Integration with Technical Quality Checker

```python
    def check_visual_quality(
        self,
        image_path: str,
        image_analysis: Dict
    ) -> Dict[str, any]:
        """
        Comprehensive visual quality check.

        Combines GPT-4 Vision analysis with computer vision metrics.

        Returns:
            Dict with visual quality assessment
        """
        results = {}
        scores = []

        # 1. GPT-4 Vision quality assessment
        vision_quality = self.vision_analyzer.analyze_quality(image_path)
        results['vision_quality'] = vision_quality
        scores.append(vision_quality['overall_quality_score'])

        # 2. Composition analysis
        composition = self.vision_analyzer.check_aesthetic_composition(image_path)
        results['composition'] = composition
        scores.append(composition['overall_composition_score'])

        # 3. Technical quality (resolution, sharpness, artifacts)
        resolution = self.check_resolution(image_path)
        results['resolution'] = resolution
        scores.append(resision['score'])

        sharpness = self.check_sharpness(image_path)
        results['sharpness'] = sharpness
        scores.append(sharpness['score'])

        artifacts = self.check_artifacts(image_path, image_analysis)
        results['artifacts'] = artifacts
        scores.append(artifacts['score'])

        # 4. Color quality
        color_analysis = image_analysis.get('colors', {})
        color_quality = self.check_color_quality(image_path, color_analysis)
        results['color_quality'] = color_quality
        scores.append(color_quality['score'])

        # Calculate overall visual quality score
        overall_score = sum(scores) / len(scores) if scores else 0

        # Determine compliance
        compliant = overall_score >= 75.0

        # Aggregate issues
        all_issues = []
        if 'primary_quality_issues' in vision_quality:
            all_issues.extend(vision_quality['primary_quality_issues'])
        if 'specific_composition_issues' in composition:
            all_issues.extend(composition['specific_composition_issues'])
        if resolution.get('issue'):
            all_issues.append(resolution['issue'])
        if sharpness.get('issue'):
            all_issues.append(sharpness['issue'])
        if artifacts.get('issue'):
            all_issues.append(artifacts['issue'])
        if color_quality.get('issue'):
            all_issues.append(color_quality['issue'])

        # Generate recommendations
        recommendations = []
        if 'improvement_suggestions' in vision_quality:
            recommendations.extend(vision_quality['improvement_suggestions'])
        if 'composition_improvements' in composition:
            recommendations.extend(composition['composition_improvements'])
        if resolution.get('recommendation'):
            recommendations.append(resolution['recommendation'])
        if sharpness.get('recommendation'):
            recommendations.append(sharpness['recommendation'])
        if artifacts.get('recommendation'):
            recommendations.append(artifacts['recommendation'])

        return {
            'compliant': compliant,
            'score': overall_score,
            'detailed_results': results,
            'issues': all_issues,
            'recommendations': recommendations,
            'quality_rating': self._get_quality_rating(overall_score)
        }

    def _get_quality_rating(self, score: float) -> str:
        """Get quality rating from score."""
        if score >= 90:
            return 'excellent'
        elif score >= 80:
            return 'good'
        elif score >= 70:
            return 'acceptable'
        elif score >= 60:
            return 'fair'
        else:
            return 'poor'
```

---

## 3. Text Quality Checks

### 3.1 Text Readability Assessment

```python
    def check_text_quality(
        self,
        image_path: str,
        image_analysis: Dict
    ) -> Dict[str, any]:
        """
        Check text overlay quality.

        Returns:
            Dict with text quality assessment
        """
        import pytesseract
        from PIL import Image

        # Get text overlay analysis
        text_overlays = image_analysis.get('text_overlays', {})
        text_regions = text_overlays.get('regions', [])

        if not text_regions:
            # No text - this is OK
            return {
                'compliant': True,
                'score': 100.0,
                'has_text': False,
                'message': 'No text overlays detected'
            }

        results = {}
        scores = []

        # 1. Text extraction and OCR quality
        try:
            img = Image.open(image_path)
            ocr_text = pytesseract.image_to_string(img)
            ocr_confidence = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

            # Average confidence
            confidences = [int(conf) for conf in ocr_confidence['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            results['ocr_quality'] = {
                'extracted_text': ocr_text.strip(),
                'avg_confidence': avg_confidence,
                'text_readable': avg_confidence > 60
            }
            scores.append(avg_confidence)

        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            results['ocr_quality'] = {
                'error': str(e),
                'text_readable': False
            }
            scores.append(50.0)

        # 2. Check each text region
        region_scores = []
        all_issues = []

        for i, region in enumerate(text_regions):
            region_result = self._check_text_region(region, i)
            region_scores.append(region_result['score'])
            all_issues.extend(region_result.get('issues', []))

        results['regions'] = text_regions
        results['region_scores'] = region_scores

        if region_scores:
            scores.append(sum(region_scores) / len(region_scores))

        # 3. Overall text ratio
        text_ratio = text_overlays.get('text_area_ratio', 0)
        if text_ratio > 0.25:  # More than 25% text
            all_issues.append(f'Text ratio too high: {text_ratio:.1%}')
            scores.append(50.0)
        else:
            scores.append(100.0)

        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else 0

        # Determine compliance
        compliant = overall_score >= 70.0 and len(
            [i for i in all_issues if 'critical' in i.lower()]
        ) == 0

        return {
            'compliant': compliant,
            'score': overall_score,
            'has_text': True,
            'text_region_count': len(text_regions),
            'text_ratio': text_ratio,
            'detailed_results': results,
            'issues': all_issues,
            'recommendations': self._get_text_recommendations(all_issues)
        }

    def _check_text_region(self, region: Dict, index: int) -> Dict[str, any]:
        """Check quality of a single text region."""
        issues = []
        score = 100.0

        text = region.get('text', '')

        # 1. Contrast check
        contrast_ratio = region.get('contrast_ratio', 0)
        min_contrast = 4.5  # WCAG AA

        if contrast_ratio < min_contrast:
            issues.append({
                'region': index,
                'type': 'low_contrast',
                'severity': 'high',
                'message': f'Text "{text[:20]}..." has low contrast (ratio: {contrast_ratio:.1f}, min: {min_contrast})'
            })
            score -= 20.0

        # 2. Font size check
        font_size = region.get('font_size', 0)
        min_size_percent = 3.0  # 3% of image height

        if font_size < min_size_percent:
            issues.append({
                'region': index,
                'type': 'small_font',
                'severity': 'medium',
                'message': f'Text "{text[:20]}..." font too small: {font_size:.1f}% (min: {min_size_percent}%)'
            })
            score -= 15.0

        # 3. Font weight check
        font_weight = region.get('font_weight', 400)
        if font_weight < 400:
            issues.append({
                'region': index,
                'type': 'light_font',
                'severity': 'low',
                'message': f'Text "{text[:20]}..." font weight too light: {font_weight}'
            })
            score -= 5.0

        # 4. Cut-off check
        is_cut_off = region.get('is_cut_off', False)
        if is_cut_off:
            issues.append({
                'region': index,
                'type': 'cut_off',
                'severity': 'critical',
                'message': f'Text "{text[:20]}..." is cut off at image edge'
            })
            score -= 30.0

        # 5. Blur check
        blur_score = region.get('blur_score', 0)
        if blur_score > 0.2:
            issues.append({
                'region': index,
                'type': 'blurry',
                'severity': 'high',
                'message': f'Text "{text[:20]}..." appears blurry (score: {blur_score:.2f})'
            })
            score -= 15.0

        # 6. Readability check
        character_count = len(text)
        if character_count > 125:  # Too much text
            issues.append({
                'region': index,
                'type': 'too_much_text',
                'severity': 'medium',
                'message': f'Text region has {character_count} characters (recommended: <125)'
            })
            score -= 10.0

        return {
            'region_index': index,
            'score': max(0.0, score),
            'issues': issues
        }

    def _get_text_recommendations(self, issues: List) -> List[str]:
        """Generate text quality recommendations."""
        recommendations = []

        for issue in issues:
            if isinstance(issue, str):
                continue

            issue_type = issue.get('type', '')
            severity = issue.get('severity', '')

            if issue_type == 'low_contrast':
                recommendations.append(
                    f"Increase text contrast - use lighter text on darker background or vice versa"
                )

            elif issue_type == 'small_font':
                recommendations.append(
                    f"Increase font size to at least 3% of image height for better readability"
                )

            elif issue_type == 'light_font':
                recommendations.append(
                    f"Use bolder font weight (600-700) for improved readability"
                )

            elif issue_type == 'cut_off':
                recommendations.append(
                    f"Adjust text placement or image size to ensure all text is fully visible"
                )

            elif issue_type == 'blurry':
                recommendations.append(
                    f"Use sharper text rendering - check export quality and font resolution"
                )

            elif issue_type == 'too_much_text':
                recommendations.append(
                    f"Reduce text content - keep messages concise and focus on key points"
                )

        # Remove duplicates
        return list(set(recommendations))
```

---

## 4. Complete Technical Quality Checker

```python
# src/meta/ad/qa/checkers/technical_quality_checker.py

class TechnicalQualityChecker:
    """Complete technical quality checker."""

    def __init__(self, config: CriteriaConfig):
        self.config = config
        self.vision_analyzer = VisionAnalyzer()

    def check(
        self,
        image_path: str,
        image_analysis: Dict
    ) -> TechnicalQualityResult:
        """
        Perform complete technical quality check.

        Returns:
            TechnicalQualityResult with all quality metrics
        """
        logger.info(f"Checking technical quality: {image_path}")

        # 1. Resolution check
        resolution = self.check_resolution(image_path)

        # 2. File size check
        file_size = self.check_file_size(image_path)

        # 3. Sharpness check
        sharpness = self.check_sharpness(image_path)

        # 4. Edge quality
        edge_quality = self.check_edge_quality(image_path)

        # 5. Artifacts check
        artifacts = self.check_artifacts(image_path, image_analysis)

        # 6. Color quality
        color_quality = self.check_color_quality(
            image_path,
            image_analysis.get('colors', {})
        )

        # 7. Visual quality (GPT-4 Vision)
        visual_quality = self.vision_analyzer.analyze_quality(image_path)

        # 8. Composition quality
        composition = self.vision_analyzer.check_aesthetic_composition(image_path)

        # 9. Text quality
        text_quality = self.check_text_quality(image_path, image_analysis)

        # Calculate overall score
        scores = [
            resolution['score'],
            sharpness['score'],
            edge_quality['score'],
            artifacts['score'],
            color_quality['score'],
            visual_quality['overall_quality_score'],
            composition['overall_composition_score']
        ]

        # Add text score if text exists
        if text_quality.get('has_text'):
            scores.append(text_quality['score'])

        overall_score = sum(scores) / len(scores) if scores else 0

        # Determine compliance
        compliant = overall_score >= self.config.min_technical_score

        # Aggregate all issues
        all_issues = []
        for result in [resolution, file_size, sharpness, edge_quality, artifacts, color_quality]:
            if result.get('issue'):
                all_issues.append(result['issue'])

        if 'primary_quality_issues' in visual_quality:
            all_issues.extend(visual_quality['primary_quality_issues'])

        if 'specific_composition_issues' in composition:
            all_issues.extend(composition['specific_composition_issues'])

        if 'issues' in text_quality:
            all_issues.extend(text_quality['issues'])

        logger.info(
            f"Technical quality check complete: Score={overall_score:.1f}, "
            f"Compliant={compliant}"
        )

        return TechnicalQualityResult(
            score=overall_score,
            compliant=compliant,
            resolution=resolution,
            sharpness=sharpness,
            edge_quality=edge_quality,
            artifacts=artifacts.get('detected', {}).get('artifacts', []),
            color_quality=color_quality,
            visual_quality=visual_quality,
            composition=composition,
            text_quality=text_quality,
            file_size=file_size,
            all_issues=all_issues,
            recommendations=self._generate_quality_recommendations({
                'resolution': resolution,
                'sharpness': sharpness,
                'artifacts': artifacts,
                'color': color_quality,
                'visual': visual_quality,
                'composition': composition,
                'text': text_quality
            })
        )

    def _generate_quality_recommendations(self, results: Dict) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []

        # Priority issues first
        if results['resolution'].get('score', 100) < 70:
            recommendations.append(
                "Increase image resolution to at least 1080x1080 for better quality"
            )

        if results['sharpness'].get('score', 100) < 70:
            recommendations.append(
                "Improve image sharpness - use in-focus source image or proper sharpening"
            )

        if results['artifacts'].get('score', 100) < 70:
            recommendations.append(
                "Reduce compression artifacts - use higher quality export settings"
            )

        if results['visual'].get('overall_quality_score', 100) < 70:
            recommendations.append(
                "Overall visual quality needs improvement - consider professional design or photography"
            )

        # Composition recommendations
        comp = results['composition']
        if not comp.get('rule_of_thirds', {}).get('followed', True):
            recommendations.append(
                "Improve composition by placing key elements on rule-of-thirds grid lines"
            )

        if comp.get('negative_space', {}).get('amount') == 'too_little':
            recommendations.append(
                "Add breathing room - reduce clutter and increase negative space"
            )

        # Color recommendations
        color_issues = results['color'].get('detected', {}).get('issues', [])
        if 'low_saturation' in color_issues:
            recommendations.append(
                "Increase color saturation for more vibrant, engaging visuals"
            )

        if 'too_dark' in color_issues:
            recommendations.append(
                "Brighten image for better visibility and impact"
            )

        # Text recommendations
        text_result = results['text']
        if text_result.get('recommendations'):
            recommendations.extend(text_result['recommendations'])

        return recommendations
```

---

## 5. Summary: Quality Checking Matrix

| Quality Dimension | Methods | Tools | Thresholds |
|---|---|---|---|
| **Resolution** | Pixel count, aspect ratio | PIL | ≥1080x1080, standard ratios |
| **Sharpness** | Laplacian variance, edge detection | OpenCV | Variance ≥100, score ≥0.3 |
| **Artifacts** | FFT high-freq analysis, noise estimation | OpenCV, NumPy | Minimal artifacts |
| **Color Quality** | Saturation, brightness, colorfulness | OpenCV (HSV, LAB) | Balanced, not extreme |
| **Composition** | Rule of thirds, balance, negative space | GPT-4 Vision | Professional composition |
| **Lighting** | Evenness, contrast, appropriateness | GPT-4 Vision | Professional lighting |
| **Text Quality** | Contrast, size, weight, OCR | pytesseract, PIL | Contrast ≥4.5:1, size ≥3% |
| **Professional Polish** | Overall aesthetic, attention to detail | GPT-4 Vision | Commercial quality |

**All checks feed into:**
- Overall quality score (0-100)
- Compliance decision (pass/fail)
- Actionable improvement recommendations
- Detailed diagnostic information

This comprehensive approach ensures every creative meets professional standards before launch!
