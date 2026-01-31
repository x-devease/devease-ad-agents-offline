"""
Pytest configuration and fixtures for tests.
"""

import os
import shutil
import tempfile
from pathlib import Path


def pytest_configure(config):
    """
    Ensure test configuration files exist before running tests.
    This is called once at the start of the test session.
    """
    # Ensure test customer config files exist
    _ensure_test_configs()


def pytest_runtest_setup(item):
    """
    Called before each test.
    Ensure test configs exist before each config test.
    """
    # If this is a config manager test, ensure configs exist
    if "test_manager.py" in str(item.fspath):
        _ensure_test_configs()


def _ensure_test_configs():
    """Create test customer config files if they don't exist."""
    test_configs = {
        "config/test_customer/meta/config.yaml": """
metadata:
  schema_version: '2.0'
  customer: test_customer
  product: Test Product
  platform: meta
  branch: US
  campaign_goal: ROAS
  generated_at: '2026-01-30'

environment: development

mining_strategy:
  winner_quantile: 0.90
  loser_quantile: 0.10
  default_profile: balanced

nano_generation_rules:
  inference_config:
    aspect_ratio: '3:4'
    batch_size: 20
    cfg_scale: 3.5
    guidance: perspective_aware
    model: nanobanana_pro
    steps: 8
  negative_prompt: cartoon, illustration, text, watermark
  prompt_slots:
    atmosphere: test environment
    product_context: test product
  prompt_template_structure: Product on table, {atmosphere}

compositing:
  light_match_mode: soft_light
  light_match_opacity: 0.25
  light_wrap_intensity: 0.3
  shadow_direction: left

text_overlay:
  template_id: test_template
  psychology_driven: false
  smart_color_enabled: false
  collision_detection_enabled: false

psychology_catalog:
  total_types: 0
  types: []

psychology_templates: []
""",
        "config/customer1/meta/config.yaml": """
# Test customer 1 configuration
environment: development
customer: customer1
platform: meta

mining_strategy:
  winner_quantile: 0.90
  loser_quantile: 0.10

nano_generation_rules:
  inference_config:
    aspect_ratio: '3:4'
    batch_size: 20

compositing:
  light_match_mode: soft_light

text_overlay:
  template_id: test_template
""",
        "config/customer2/google/config.yaml": """
# Test customer 2 configuration
environment: development
customer: customer2
platform: google

mining_strategy:
  winner_quantile: 0.90
  loser_quantile: 0.10

nano_generation_rules:
  inference_config:
    aspect_ratio: '3:4'
    batch_size: 20

compositing:
  light_match_mode: soft_light

text_overlay:
  template_id: test_template
""",
    }

    for config_path, config_content in test_configs.items():
        config_file = Path(config_path)
        if not config_file.exists():
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_file.write_text(config_content.strip() + "\n")
