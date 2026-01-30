"""
V1.8 Pipeline Orchestrator

Coordinates all 3 stages of the Ad Miner V1.8 pipeline:
- Stage 0: Mining Strategy Selector
- Stage 1: The Miner (strict context mining)
- Stage 2: The Synthesizer (combinatorial logic)
- Stage 3: The Upscaler (multi-workflow CoT expansion)
"""
import logging
import yaml
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

from .config import MiningStrategySelector
from .stages.miner import AdMiner
from .stages.synthesizer import CombinatorialSynthesizer
from .stages.upscaler import CoTUpscaler

logger = logging.getLogger(__name__)


class V18Pipeline:
    """
    V1.8 Ad Miner Pipeline Orchestrator.

    End-to-end pipeline that takes customer config and creative features,
    then outputs a master blueprint for ad generation.
    """

    def __init__(
        self,
        config_root: Path,
        llm_client: Optional[Any] = None
    ):
        """
        Initialize V1.8 pipeline.

        Args:
            config_root: Root path for customer config files
            llm_client: LLM client for CoT expansion (optional)
        """
        self.config_root = Path(config_root)

        # Initialize Mining Strategy Selector
        self.strategy_selector = MiningStrategySelector(config_root)

        # Initialize stages
        self.miner = None  # Will be initialized with input_config
        self.synthesizer = CombinatorialSynthesizer(min_confidence=0.8)
        self.upscaler = None  # Will be initialized with fidelity_config

        # LLM client
        self.llm_client = llm_client

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

    def load_input_config(self, input_config_path: Path) -> dict:
        """
        Load input configuration (campaign-specific).

        Args:
            input_config_path: Path to input_config.yaml

        Returns:
            Input config dict
        """
        with open(input_config_path) as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded input config from {input_config_path}")
        return config

    def load_data(self, data_path: Path) -> pd.DataFrame:
        """
        Load creative features data.

        Args:
            data_path: Path to CSV file

        Returns:
            DataFrame with creative features
        """
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} creative features from {data_path}")
        return df

    def run(
        self,
        customer_id: str,
        input_config: dict,
        df: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> Dict:
        """
        Run complete V1.8 pipeline.

        Args:
            customer_id: Customer identifier
            input_config: Input configuration (campaign-specific)
            df: Creative features DataFrame
            output_path: Optional path to save master blueprint

        Returns:
            Complete master blueprint dict
        """
        logger.info("="*60)
        logger.info("Starting Ad Miner V1.8 Pipeline")
        logger.info("="*60)

        # Extract platform from input_config
        platform = input_config.get("customer_context", {}).get("platform", "meta")

        # Load customer config
        customer_config = self.load_customer_config(customer_id, platform)

        # Initialize miner with strategy selector
        self.miner = AdMiner(
            input_config=input_config,
            strategy_selector=self.strategy_selector,
            customer_config=customer_config
        )

        # Initialize upscaler if LLM client provided
        if self.llm_client:
            fidelity_config = customer_config.get("fidelity_config", {})
            brand_guidelines = customer_config.get("brand_guidelines", {})

            # Workflow templates are now in customer config file (Part J)
            workflow_templates = customer_config.get("workflow_templates", {})

            self.upscaler = CoTUpscaler(
                llm_client=self.llm_client,
                workflow_templates=workflow_templates,
                fidelity_config=fidelity_config,
                brand_guidelines=brand_guidelines
            )

        # ====================
        # Stage 0: Strategy Selection (implicit in miner)
        # ====================

        # ====================
        # Stage 1: The Miner
        # ====================
        logger.info("\n" + "="*60)
        logger.info("STAGE 1: THE MINER")
        logger.info("="*60)

        winner_quantile, winners, losers, raw_tags = self.miner.run(df)

        # ====================
        # Stage 2: The Synthesizer
        # ====================
        logger.info("\n" + "="*60)
        logger.info("STAGE 2: THE SYNTHESIZER")
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
        # Stage 3: The Upscaler (if LLM client available)
        # ====================
        logger.info("\n" + "="*60)
        logger.info("STAGE 3: THE UPSCALER")
        logger.info("="*60)

        if self.upscaler:
            # Build context for workflow selection
            context = {
                "customer_id": customer_id,
                "product_type": input_config.get("customer_context", {}).get("product", ""),
                "campaign_goal": input_config.get("campaign_context", {}).get("execution_mode", ""),
                "daily_budget_cents": input_config.get("campaign_context", {}).get("daily_budget_cents", 0),
                "product_margin": "medium",  # TODO: Get from config
                "brand_maturity": "established" if customer_config.get("brand_guidelines", {}).get("established") else "growing",
                "mining_strictness": f"Top {(1-winner_quantile)*100:.0f}%",
                "brand_name": customer_config.get("brand_guidelines", {}).get("brand_name", ""),
                "brand_guidelines": customer_config.get("brand_guidelines", {}),
                "color_system": customer_config.get("brand_guidelines", {}).get("color_system", {}),
                "primary_colors": customer_config.get("brand_guidelines", {}).get("color_system", {}).get("primary_colors", []),
                "secondary_colors": customer_config.get("brand_guidelines", {}).get("color_system", {}).get("secondary_colors", []),
            }

            master_blueprint = self.upscaler.run(
                locked_combinations=locked_combinations,
                losers_df=losers,
                raw_tags=raw_tags,
                context=context
            )
        else:
            # No LLM client - create basic blueprint without CoT expansion
            logger.warning("No LLM client provided. Creating basic blueprint without CoT expansion.")
            master_blueprint = {
                "meta_info": {
                    "recipe_id": "recipe_basic_v1.8",
                    "mining_strictness": f"Top {(1-winner_quantile)*100:.0f}%",
                    "fidelity_expanded": False,
                },
                "strategy_rationale": locked_combinations,
                "nano_generation_rules": {
                    "negative_prompt": ["cgi", "3d render", "cartoon", "blurry"],
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
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)

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


def create_pipeline(
    config_root: Path = None,
    llm_client: Optional[Any] = None
) -> V18Pipeline:
    """
    Factory function to create V1.8 pipeline.

    Args:
        config_root: Root path for customer configs (default: config)
        llm_client: Optional LLM client for CoT expansion

    Returns:
        Initialized V18Pipeline instance
    """
    if config_root is None:
        config_root = Path("config")

    return V18Pipeline(
        config_root=config_root,
        llm_client=llm_client
    )
