"""
Main pipeline for Ad Reviewer (Visual QA & Risk Matrix).

This module provides the VisualQAMatrix orchestrator that runs
the 4-Guard quality assurance pipeline.
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from .schemas.audit_report import AuditReport, GuardStatus
from .guards.geometric_guard import GeometricGuard
from .guards.aesthetic_guard import AestheticGuard
from .guards.cultural_guard import CulturalGuard
from .guards.performance_guard import PerformanceGuard
from .vlms.base import VLMClient
from .vlms.openai_vlm import OpenAIVLM


class VisualQAMatrix:
    """
    Main orchestrator for the 4-Guard Visual QA system.

    This class implements a funnel-based quality assurance pipeline that
    progressively validates generated ad images through four guards:
    1. GeometricGuard - Product integrity (CPU-local)
    2. AestheticGuard - Visual quality (VLM)
    3. CulturalGuard - Compliance (VLM)
    4. PerformanceGuard - Optimization scoring (VLM)

    The pipeline uses fail-fast behavior: any guard failure terminates
    execution and returns a FAIL report.
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        vlm_client: Optional[VLMClient] = None
    ):
        """
        Initialize the Visual QA pipeline.

        Args:
            config_path: Path to customer config YAML (config/{customer}/{platform}/config.yaml)
            vlm_client: Optional VLM client (auto-created if None)
        """
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.vlm_client = vlm_client or self._create_vlm_client()

        # Initialize guards
        self.geometric_guard = GeometricGuard(
            tolerance=self.config.get('geometry', {}).get('tolerance', 0.02),
            min_features=self.config.get('geometry', {}).get('min_features', 10),
            fallback=self.config.get('geometry', {}).get('fallback_to_contour', True),
            contour_tolerance=self.config.get('geometry', {}).get('contour_tolerance', 0.05)
        )

        self.aesthetic_guard = AestheticGuard(
            vlm_client=self.vlm_client,
            min_score=self.config.get('aesthetics', {}).get('min_score', 7.0)
        )

        self.cultural_guard = CulturalGuard(
            vlm_client=self.vlm_client,
            target_region=self.config.get('cultural', {}).get('target_region', 'Global'),
            risk_threshold=self.config.get('cultural', {}).get('risk_threshold', 'HIGH'),
            custom_rules=self.config.get('cultural', {}).get('custom_rules', [])
        )

        weights = self.config.get('performance', {}).get('psychology_weight', 0.40)
        self.performance_guard = PerformanceGuard(
            vlm_client=self.vlm_client,
            weights={
                'psychology': self.config.get('performance', {}).get('psychology_weight', 0.40),
                'saliency': self.config.get('performance', {}).get('saliency_weight', 0.30),
                'consistency': self.config.get('performance', {}).get('consistency_weight', 0.30)
            }
        )

        # Pipeline settings
        self.stop_on_first_fail = self.config.get('pipeline', {}).get('stop_on_first_fail', True)
        self.log_all_checks = self.config.get('pipeline', {}).get('log_all_checks', True)

    def audit(
        self,
        candidate_image_path: Union[str, Path],
        product_image_path: Union[str, Path],
        blueprint: Dict[str, Any]
    ) -> AuditReport:
        """
        Run complete 4-guard audit on a generated image.

        Args:
            candidate_image_path: Path to generated ad image
            product_image_path: Path to original product image
            blueprint: Blueprint dict containing strategy and psychology settings

        Returns:
            AuditReport with complete results and PASS/FAIL status
        """
        start_time = time.time()
        api_calls = 0

        # Initialize report with basic metadata
        report = AuditReport(
            session_id="",  # Will be set by audit_session
            prompt_id="",   # Will be set by audit_session
            image_id="",    # Will be set by audit_session
            image_path=str(candidate_image_path),
            product_image_path=str(product_image_path),
            generation_model="",  # Will be set by audit_session
            blueprint_id=blueprint.get('id', 'unknown'),
            psychology_driver=blueprint.get('strategy_rationale', {}).get('psychology_driver'),
            timestamp=str(time.time())
        )

        try:
            # -------------------------------------------------
            # GUARD 1: Geometric (CPU-local, fast)
            # -------------------------------------------------
            if self.config.get('geometry', {}).get('enabled', True):
                geo_result = self.geometric_guard.check(
                    raw_product_path=str(product_image_path),
                    candidate_path=str(candidate_image_path)
                )

                report.geometric = geo_result

                if geo_result.status == GuardStatus.FAIL and self.stop_on_first_fail:
                    return self._fail_report(
                        report,
                        fail_guard="geometric",
                        fail_code=geo_result.reasoning,
                        start_time=start_time,
                        api_calls=api_calls
                    )

            # -------------------------------------------------
            # GUARD 2: Aesthetic (VLM, checks quality)
            # -------------------------------------------------
            if self.config.get('aesthetics', {}).get('enabled', True):
                aes_result = self.aesthetic_guard.check(
                    image_path=str(candidate_image_path),
                    negative_prompts=blueprint.get(
                        'nano_generation_rules', {}
                    ).get('negative_prompt', [])
                )
                api_calls += 1
                report.aesthetic = aes_result

                if aes_result.status == GuardStatus.FAIL and self.stop_on_first_fail:
                    return self._fail_report(
                        report,
                        fail_guard="aesthetic",
                        fail_code=aes_result.reasoning,
                        start_time=start_time,
                        api_calls=api_calls
                    )

            # -------------------------------------------------
            # GUARD 3: Cultural (VLM, checks compliance)
            # -------------------------------------------------
            if self.config.get('cultural', {}).get('enabled', True):
                cult_result = self.cultural_guard.check(
                    image_path=str(candidate_image_path)
                )
                api_calls += 1
                report.cultural = cult_result

                if cult_result.status == GuardStatus.FAIL and self.stop_on_first_fail:
                    return self._fail_report(
                        report,
                        fail_guard="cultural",
                        fail_code=cult_result.reasoning,
                        start_time=start_time,
                        api_calls=api_calls
                    )

            # -------------------------------------------------
            # GUARD 4: Performance (VLM, scoring for ranking)
            # -------------------------------------------------
            if self.config.get('performance', {}).get('enabled', True):
                perf_result = self.performance_guard.score(
                    image_path=str(candidate_image_path),
                    psychology_goal=blueprint.get(
                        'strategy_rationale', {}
                    ).get('psychology_driver', 'engagement')
                )
                api_calls += 1
                report.performance = perf_result
                report.performance_score = perf_result.overall_score

            # -------------------------------------------------
            # ALL GUARDS PASSED
            # -------------------------------------------------
            report.status = GuardStatus.PASS
            report.total_execution_time_ms = (time.time() - start_time) * 1000
            report.api_calls_count = api_calls

            return report

        except Exception as e:
            # Unexpected error during audit
            return self._fail_report(
                report,
                fail_guard="pipeline",
                fail_code=f"INTERNAL_ERROR: {str(e)}",
                start_time=start_time,
                api_calls=api_calls
            )

    def audit_session(
        self,
        session_path: Union[str, Path],
        blueprint: Optional[Dict[str, Any]] = None
    ) -> List[AuditReport]:
        """
        Audit all images from a generator session.

        This is the primary integration method with Module 2 (Generator).
        It loads the session.json and processes all generated images.

        Args:
            session_path: Path to generator session.json
            blueprint: Optional blueprint dict. If None, loads from config.

        Returns:
            List of AuditReports, one per generated image
        """
        # Load generator session
        session_path = Path(session_path)
        with open(session_path, 'r') as f:
            session = json.load(f)

        # Extract metadata
        session_id = session.get('session_id', 'unknown')
        prompts = session.get('prompts', [])

        # If blueprint not provided, try to load from formula path
        if blueprint is None:
            formula_path = session.get('metadata', {}).get('formula_path')
            if formula_path:
                blueprint = self._load_blueprint_from_formula(formula_path)
            else:
                blueprint = self._create_default_blueprint()

        all_reports = []

        # Process each prompt and its images
        for prompt_record in prompts:
            prompt_id = prompt_record.get('prompt_id', 'unknown')
            product_image_path = prompt_record.get(
                'product_context', {}
            ).get('source_image_path')

            if not product_image_path:
                # Skip if no product image available
                continue

            images = prompt_record.get('images', [])

            for image_record in images:
                image_id = image_record.get('image_id', 'unknown')
                image_path = image_record.get('image_path')
                generation_model = image_record.get('generation_model', 'unknown')

                if not image_path or not Path(image_path).exists():
                    # Skip missing images
                    continue

                # Run audit
                report = self.audit(
                    candidate_image_path=image_path,
                    product_image_path=product_image_path,
                    blueprint=blueprint
                )

                # Override with generator metadata
                report.session_id = session_id
                report.prompt_id = prompt_id
                report.image_id = image_id
                report.generation_model = generation_model

                all_reports.append(report)

        return all_reports

    def _fail_report(
        self,
        report: AuditReport,
        fail_guard: str,
        fail_code: str,
        start_time: float,
        api_calls: int
    ) -> AuditReport:
        """Convert report to FAIL status."""
        report.status = GuardStatus.FAIL
        report.fail_guard = fail_guard
        report.fail_code = fail_code
        report.fail_reason = fail_code
        report.total_execution_time_ms = (time.time() - start_time) * 1000
        report.api_calls_count = api_calls
        return report

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load and validate configuration from YAML."""
        # Use existing config loader
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

        try:
            from src.meta.ad.miner.utils.config_loader import ConfigLoader
            loader = ConfigLoader()
            full_config = loader.load(str(config_path))
            return full_config.get('qa_risk_matrix', {})
        except Exception:
            # If config loader fails, return empty dict
            return {}

    def _create_vlm_client(self) -> VLMClient:
        """Create VLM client based on config."""
        model = self.config.get('aesthetics', {}).get('model', 'gpt-4o-mini')

        if model.startswith('gpt'):
            return OpenAIVLM(model=model)
        else:
            raise ValueError(f"Unsupported VLM model: {model}")

    def _load_blueprint_from_formula(
        self,
        formula_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Load blueprint from generator's formula/recommendations file."""
        # Placeholder implementation
        # In production, this would parse the formula file
        return self._create_default_blueprint()

    def _create_default_blueprint(self) -> Dict[str, Any]:
        """Create default blueprint if none available."""
        return {
            'id': 'default',
            'strategy_rationale': {
                'psychology_driver': 'engagement'
            },
            'nano_generation_rules': {
                'negative_prompt': []
            }
        }
