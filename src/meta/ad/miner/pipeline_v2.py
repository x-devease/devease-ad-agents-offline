"""
V2.0 Pipeline Orchestrator with Psych-Composer

Coordinates all 4 stages of the Ad Miner V2.0 pipeline:
- Stage 1: The Miner V2 (visual + psychological extraction)
- Stage 2: The Synthesizer (combinatorial logic)
- Stage 2.5: The Psych-Composer (psychological filtering) - NEW
- Stage 3: The Upscaler V2 (psychology-aware CoT expansion)
"""
import logging
import yaml
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

from .config import MiningStrategySelector
from .stages.miner_v2 import AdMinerV2
from .stages.synthesizer import CombinatorialSynthesizer
from .stages.psych_composer import PsychComposer, create_psych_composer
from .stages.upscaler_v2 import CoTUpscalerV2

logger = logging.getLogger(__name__)


class V20Pipeline:
    """
    V2.0 Ad Miner Pipeline Orchestrator with Psychological Intelligence.

    End-to-end pipeline that takes customer config and creative features,
    then outputs a psychology-driven master blueprint for ad generation.
    """

    def __init__(
        self,
        config_root: Path,
        llm_client: Optional[Any] = None,
        vlm_client: Optional[Any] = None
    ):
        """
        Initialize V2.0 pipeline.

        Args:
            config_root: Root path for customer config files
            llm_client: LLM client for CoT expansion (optional)
            vlm_client: VLM client for psychology extraction (optional)
        """
        self.config_root = Path(config_root)

        # Initialize Mining Strategy Selector
        self.strategy_selector = MiningStrategySelector(config_root)

        # Initialize stages
        self.miner = None  # Will be initialized with input_config
        self.synthesizer = CombinatorialSynthesizer(min_confidence=0.8)
        self.psych_composer = None  # Will be initialized with target_psychology
        self.upscaler = None  # Will be initialized with fidelity_config

        # LLM/VLM clients
        self.llm_client = llm_client
        self.vlm_client = vlm_client

    def load_customer_config(self, customer_id: str, platform: str) -> dict:
        """
        Load customer configuration.

        Args:
            customer_id: Customer identifier
            platform: Platform identifier (meta, tiktok, google)

        Returns:
            Customer config dict
        """
        config_path = self.config_root / customer_id / platform / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Customer config not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded customer config for '{customer_id}' on platform '{platform}'")
        return config

    def run(
        self,
        customer_id: str,
        input_config: dict,
        df: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> Dict:
        """
        Run complete V2.0 pipeline with psychology.

        Args:
            customer_id: Customer identifier
            input_config: Input configuration (campaign-specific)
            df: Creative features DataFrame
            output_path: Optional path to save master blueprint

        Returns:
            Complete psychology-driven master blueprint dict
        """
        logger.info("="*60)
        logger.info("Starting Ad Miner V2.0 Pipeline (Psych-Composer)")
        logger.info("="*60)

        # Extract platform from input_config
        platform = input_config.get("customer_context", {}).get("platform", "meta")

        # Load customer config
        customer_config = self.load_customer_config(customer_id, platform)

        # ====================
        # Stage 1: Enhanced Miner (Visual + Psychology)
        # ====================
        logger.info("\n" + "="*60)
        logger.info("STAGE 1: THE MINER V2 (with Psychology Tagging)")
        logger.info("="*60)

        # Initialize enhanced miner
        self.miner = AdMinerV2(
            input_config=input_config,
            strategy_selector=self.strategy_selector,
            customer_config=customer_config,
            vlm_client=self.vlm_client
        )

        winner_quantile, winners, losers, raw_tags = self.miner.run(df)

        # ====================
        # Stage 2: Visual Synthesizer (V1.8)
        # ====================
        logger.info("\n" + "="*60)
        logger.info("STAGE 2: THE SYNTHESIZER (Visual)")
        logger.info("="*60)

        locked_combinations = self.synthesizer.run(winners, raw_tags)

        if not locked_combinations:
            logger.warning("No locked combinations found. Skipping Stage 3.")
            return {
                "error": "No locked combinations found",
                "winners_count": len(winners),
                "raw_tags": raw_tags
            }

        # ====================
        # Stage 2.5: Psych-Composer (NEW)
        # ====================
        logger.info("\n" + "="*60)
        logger.info("STAGE 2.5: THE PSYCH-COMPOSER")
        logger.info("="*60)

        # Determine target psychology
        psychology_config = input_config.get("psychology_config", {})
        psychology_mode = psychology_config.get("psychology_mode", "auto")

        if psychology_mode == "auto":
            # Auto-detect from winners
            psych_detection = raw_tags.get("_target_psychology_detection", {})
            target_psychology = psych_detection.get("target_psychology", "Trust_Authority")
            logger.info(f"Auto-detected psychology: {target_psychology}")
        else:
            # Use manual specification
            target_psychology = input_config.get("customer_context", {}).get(
                "target_psychology",
                "Trust_Authority"
            )
            logger.info(f"Manual psychology: {target_psychology}")

        # Get psychology mappings from customer config
        psychology_mappings = customer_config.get("psychology_mappings", {})

        # Initialize Psych-Composer
        self.psych_composer = create_psych_composer(
            psychology_mappings=psychology_mappings if psychology_mappings else None,
            target_psychology=target_psychology,
            strictness=psychology_config.get("psychology_strictness", "strict")
        )

        # Apply psychological filtering
        psych_validated_combinations = self.psych_composer.compose(locked_combinations)

        if not psych_validated_combinations:
            logger.warning("No psychologically validated combinations found.")
            return {
                "error": "No psych-validated combinations",
                "target_psychology": target_psychology,
                "visual_combinations_count": len(locked_combinations)
            }

        # ====================
        # Stage 3: Psychology-Aware Upscaler
        # ====================
        logger.info("\n" + "="*60)
        logger.info("STAGE 3: THE UPSCALER V2 (Psychology-Aware)")
        logger.info("="*60)

        # Initialize V2 upscaler
        if self.llm_client:
            fidelity_config = customer_config.get("fidelity_config", {})
            brand_guidelines = customer_config.get("brand_guidelines", {})
            workflow_templates = customer_config.get("workflow_templates", {})

            self.upscaler = CoTUpscalerV2(
                llm_client=self.llm_client,
                workflow_templates=workflow_templates,
                fidelity_config=fidelity_config,
                brand_guidelines=brand_guidelines
            )

            # Build context
            context = {
                "customer_id": customer_id,
                "product_type": input_config.get("customer_context", {}).get("product", ""),
                "campaign_goal": input_config.get("campaign_context", {}).get("execution_mode", ""),
                "daily_budget_cents": input_config.get("campaign_context", {}).get("daily_budget_cents", 0),
                "product_margin": "medium",
                "brand_maturity": "established" if customer_config.get("brand_guidelines", {}).get("established") else "growing",
                "mining_strictness": f"Top {(1-winner_quantile)*100:.0f}%",
                "brand_name": customer_config.get("brand_guidelines", {}).get("brand_name", ""),
                "brand_guidelines": customer_config.get("brand_guidelines", {}),
                "color_system": customer_config.get("brand_guidelines", {}).get("color_system", {}),
                "primary_colors": customer_config.get("brand_guidelines", {}).get("color_system", {}).get("primary_colors", []),
                "secondary_colors": customer_config.get("brand_guidelines", {}).get("color_system", {}).get("secondary_colors", []),
                "target_psychology": target_psychology,
                "psychology_mode": psychology_mode
            }

            master_blueprint = self.upscaler.build_psychology_aware_blueprint(
                locked_combinations=psych_validated_combinations,
                losers_df=losers,
                raw_tags=raw_tags,
                context=context,
                target_psychology=target_psychology
            )

            # Add additional metadata
            master_blueprint["meta_info"]["psychology_confidence"] = \
                raw_tags.get("_target_psychology_detection", {}).get("confidence", 0.0)

            if raw_tags.get("_target_psychology_detection"):
                master_blueprint["meta_info"]["psychology_distribution"] = \
                    raw_tags["_target_psychology_detection"]["distribution"]

        else:
            # No LLM client - create basic blueprint without CoT expansion
            logger.warning("No LLM client provided. Creating basic blueprint without CoT expansion.")

            # Get first validated combination for basic blueprint
            first_combo_key = list(psych_validated_combinations.keys())[0]
            first_combo = psych_validated_combinations[first_combo_key]

            master_blueprint = {
                "meta_info": {
                    "recipe_id": "recipe_psych_v2.0_basic",
                    "mining_strictness": f"Top {(1-winner_quantile)*100:.0f}%",
                    "target_psychology": target_psychology,
                    "fidelity_expanded": False,
                },
                "strategy_rationale": {
                    "psychology_driver": target_psychology,
                    "combinations": {
                        first_combo_key: first_combo
                    }
                },
                "nano_generation_rules": {
                    "prompt_template_structure": f"Psychology: {target_psychology}",
                    "prompt_slots": {
                        "quality_headers": self.upscaler.PSYCHOLOGY_QUALITY_HEADERS.get(
                            target_psychology,
                            "Raw photo, 8k uhd, masterpiece"
                        ) if self.upscader else "Raw photo, 8k uhd, masterpiece"
                    },
                    "negative_prompt": self.upscaler.PSYCHOLOGY_NEGATIVE_PROMPTS.get(
                        target_psychology,
                        ["cgi", "3d render", "cartoon", "blurry"]
                    ) if self.upscader else ["cgi", "3d render", "cartoon", "blurry"],
                    "inference_config": {
                        "steps": 8,
                        "cfg_scale": 1.5,
                        "width": 1024,
                        "height": 1024
                    }
                }
            }

        # ====================
        # Save Output
        # ====================
        if output_path:
            self.save_blueprint(master_blueprint, output_path)

        logger.info("\n" + "="*60)
        logger.info("V2.0 PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Target Psychology: {target_psychology}")
        logger.info(f"Validated Combinations: {len(psych_validated_combinations)}")

        return master_blueprint

    def save_blueprint(self, blueprint: dict, output_path: Path):
        """
        Save master blueprint to YAML file.

        Args:
            blueprint: Master blueprint dict
            output_path: Path to save blueprint
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(blueprint, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved master blueprint to {output_path}")


def create_pipeline_v2(
    config_root: Path = None,
    llm_client: Optional[Any] = None,
    vlm_client: Optional[Any] = None
) -> V20Pipeline:
    """
    Factory function to create V2.0 pipeline.

    Args:
        config_root: Root path for customer configs (default: config)
        llm_client: Optional LLM client for CoT expansion
        vlm_client: Optional VLM client for psychology extraction

    Returns:
        Initialized V20Pipeline instance
    """
    if config_root is None:
        config_root = Path("config")

    return V20Pipeline(
        config_root=config_root,
        llm_client=llm_client,
        vlm_client=vlm_client
    )
