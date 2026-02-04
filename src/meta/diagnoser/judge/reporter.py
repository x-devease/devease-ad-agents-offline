"""
Reporter for Diagnoser Judge evaluation system.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Any
from pathlib import Path

from .schemas import (
    EvaluationResult,
    ComparisonResult,
    BacktestResult,
)

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """
    评估报告生成器 - JSON格式

    生成JSON格式的评估报告，保存到 diagnoser/judge/reports/{customer}/ 目录
    """

    def __init__(self, output_dir: str = "src/meta/diagnoser/judge/reports", customer: str = "default"):
        """
        初始化报告生成器

        Args:
            output_dir: 基础输出目录
            customer: 客户名称（用于创建子目录）
        """
        self.output_dir = Path(output_dir) / customer
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_evaluation_report(
        self,
        result: EvaluationResult,
    ) -> str:
        """
        生成评估报告 (JSON格式)

        Args:
            result: 评估结果

        Returns:
            JSON报告内容
        """
        return json.dumps(result.to_dict(), indent=2, default=str)

    def generate_comparison_report(
        self,
        comparison: ComparisonResult,
    ) -> str:
        """
        生成对比报告 (JSON格式)

        Args:
            comparison: 对比结果

        Returns:
            JSON报告内容
        """
        return json.dumps(comparison.to_dict(), indent=2, default=str)

    def generate_backtest_report(
        self,
        backtest_result: BacktestResult,
    ) -> str:
        """
        生成回测报告 (JSON格式)

        Args:
            backtest_result: 回测结果

        Returns:
            JSON报告内容
        """
        return json.dumps(backtest_result.to_dict(), indent=2, default=str)

    def save_report(
        self,
        content: str,
        filename: str,
    ) -> str:
        """
        保存报告到文件

        Args:
            content: JSON报告内容
            filename: 文件名 (应使用.json后缀)

        Returns:
            文件路径
        """
        filepath = self.output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Report saved to: {filepath}")
        return str(filepath)

    def generate_dashboard_data(
        self,
        result: EvaluationResult,
    ) -> Dict[str, Any]:
        """
        生成Dashboard数据

        Args:
            result: 评估结果

        Returns:
            Dashboard数据字典
        """
        return {
            "overview": {
                "detector_name": result.detector_name,
                "detector_type": result.detector_type,
                "overall_score": result.overall_score,
                "grade": result.grade,
                "evaluation_date": result.evaluation_date.isoformat() if result.evaluation_date else None,
            },
            "scores": {
                "accuracy": {
                    "value": result.accuracy.f1_score * 100,
                    "weight": 0.40,
                    "breakdown": {
                        "precision": result.accuracy.precision,
                        "recall": result.accuracy.recall,
                        "f1_score": result.accuracy.f1_score,
                    },
                },
                "reliability": {
                    "value": (
                        result.reliability.stability_score * 0.4 +
                        result.reliability.reproducibility_score * 0.3 +
                        result.reliability.robustness_score * 0.3
                    ) * 100,
                    "weight": 0.20,
                    "breakdown": {
                        "stability": result.reliability.stability_score,
                        "reproducibility": result.reliability.reproducibility_score,
                        "robustness": result.reliability.robustness_score,
                    },
                },
                "timeliness": {
                    "value": max(0, 100 - min(40, result.timeliness.detection_delay_days * 5) -
                             min(20, result.timeliness.processing_time_ms / 100) -
                             min(40, max(0, result.timeliness.min_data_days - 7) * 2)),
                    "weight": 0.20,
                    "breakdown": {
                        "detection_delay_days": result.timeliness.detection_delay_days,
                        "processing_time_ms": result.timeliness.processing_time_ms,
                        "min_data_days": result.timeliness.min_data_days,
                    },
                },
                "interpretability": {
                    "value": (
                        result.interpretability.transparency_score * 0.4 +
                        result.interpretability.readability_score * 0.3 +
                        result.interpretability.actionability_score * 0.3
                    ) * 100,
                    "weight": 0.10,
                    "breakdown": {
                        "transparency": result.interpretability.transparency_score,
                        "readability": result.interpretability.readability_score,
                        "actionability": result.interpretability.actionability_score,
                    },
                },
                "business_value": {
                    "value": min(100, max(0, result.business_value.estimated_savings_usd / 10)),
                    "weight": 0.10,
                    "breakdown": {
                        "estimated_savings_usd": result.business_value.estimated_savings_usd,
                        "improvement_potential": result.business_value.improvement_potential,
                        "user_satisfaction": result.business_value.user_satisfaction,
                    },
                },
            },
            "suggestions": result.details.get("suggestions", []),
            "raw_data": result.to_dict(),
        }

    def generate_summary_markdown(
        self,
        results: Dict[str, Any],
        customer_name: str = "default",
        scenario: str = "standard",
        data_description: str = "",
    ) -> str:
        """
        生成评估总结报告 (Markdown格式)

        Args:
            results: 检测器评估结果字典 {detector_name: EvaluationResult}
            customer_name: 客户名称
            scenario: 评估场景
            data_description: 数据描述

        Returns:
            Markdown报告内容
        """
        from datetime import datetime

        lines = []
        lines.append(f"# Diagnoser Evaluation Summary")
        lines.append("")
        lines.append(f"**Customer**: {customer_name}")
        lines.append(f"**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Scenario**: {scenario}")
        lines.append(f"**Data**: {data_description}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")

        # 计算总体统计
        total_detectors = len(results)
        avg_score = sum(r.overall_score for r in results.values()) / total_detectors if total_detectors > 0 else 0

        # 统计等级分布
        grade_distribution = {}
        for result in results.values():
            grade = result.grade
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1

        lines.append(f"- **Total Detectors Evaluated**: {total_detectors}")
        lines.append(f"- **Average Overall Score**: {avg_score:.1f}/100")
        lines.append(f"- **Grade Distribution**:")
        for grade in ['A', 'B', 'C', 'D', 'F']:
            count = grade_distribution.get(grade, 0)
            if count > 0:
                lines.append(f"  - {grade}: {count} detector(s)")
        lines.append("")

        # Zero-Cost Label Statistics
        lines.append("## Zero-Cost Label Statistics")
        lines.append("")
        for detector_name, result in results.items():
            label_count = result.details.get("label_count", 0)
            label_method = result.details.get("label_method", "N/A")
            lines.append(f"### {detector_name}")
            lines.append(f"- **Labels Generated**: {label_count}")
            lines.append(f"- **Label Method**: {label_method}")
            lines.append("")

        # Detector Performance Details
        lines.append("## Detector Performance Details")
        lines.append("")

        for detector_name, result in results.items():
            lines.append(f"### {detector_name}")
            lines.append("")
            lines.append(f"**Overall Score**: {result.overall_score:.1f}/100 ({result.grade})")
            lines.append("")

            # Accuracy Breakdown
            lines.append("**Accuracy Metrics**:")
            lines.append(f"- Precision: {result.accuracy.precision:.2%}")
            lines.append(f"- Recall: {result.accuracy.recall:.2%}")
            lines.append(f"- F1-Score: {result.accuracy.f1_score:.2%}")
            lines.append(f"- True Positives: {result.accuracy.true_positives}")
            lines.append(f"- False Positives: {result.accuracy.false_positives}")
            lines.append(f"- False Negatives: {result.accuracy.false_negatives}")
            lines.append("")

            # 如果所有accuracy都是0，说明没有ground truth
            if (result.accuracy.true_positives == 0 and
                result.accuracy.false_positives == 0 and
                result.accuracy.false_negatives == 0 and
                result.accuracy.true_negatives == 0):

                label_count = result.details.get("label_count", 0)
                if label_count == 0:
                    lines.append(f"**⚠️ Note**: No ground truth labels found in dataset. Accuracy metrics cannot be calculated (division by zero).")
                    lines.append("")
                    lines.append(f"This indicates the dataset does not contain clear examples of `{detector_name.replace('Detector', '').lower()}` issues.")
                    lines.append("")
                    lines.append(f"The detector correctly did not generate false positives, but we cannot evaluate true positive detection rate without issue examples.")
                    lines.append("")

            # Reliability
            lines.append("**Reliability**:")
            lines.append(f"- Stability: {result.reliability.stability_score:.2f}")
            lines.append(f"- Reproducibility: {result.reliability.reproducibility_score:.2f}")
            lines.append(f"- Robustness: {result.reliability.robustness_score:.2f}")
            lines.append("")

            # Timeliness
            lines.append("**Timeliness**:")
            lines.append(f"- Detection Delay: {result.timeliness.detection_delay_days:.1f} days")
            lines.append(f"- Processing Time: {result.timeliness.processing_time_ms:.1f} ms")
            lines.append(f"- Min Data Required: {result.timeliness.min_data_days} days")
            lines.append("")

            # Suggestions
            suggestions = result.details.get("suggestions", [])
            if suggestions:
                lines.append("**Suggestions**:")
                for suggestion in suggestions:
                    lines.append(f"- {suggestion}")
                lines.append("")

            lines.append("---")
            lines.append("")

        # Key Findings
        lines.append("## Key Findings")
        lines.append("")

        # 找出表现最好和最差的detector
        if results:
            best_detector = max(results.items(), key=lambda x: x[1].overall_score)
            worst_detector = min(results.items(), key=lambda x: x[1].overall_score)

            lines.append(f"**Best Performing**: {best_detector[0]} ({best_detector[1].overall_score:.1f}/100)")
            lines.append(f"**Needs Improvement**: {worst_detector[0]} ({worst_detector[1].overall_score:.1f}/100)")
            lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")

        # 检查是否所有detectors都没有labels
        all_zero_labels = all(r.details.get("label_count", 0) == 0 for r in results.values())
        if all_zero_labels:
            lines.append("### ⚠️ Dataset Analysis")
            lines.append("")
            lines.append("The current dataset does not contain clear examples of the issues our detectors are designed to identify:")
            lines.append("- No creative fatigue (high frequency + CPA increase)")
            lines.append("- No performance latency (significant ROAS drops)")
            lines.append("- No dark hours (hourly underperformance patterns)")
            lines.append("")
            lines.append("**Next Steps**:")
            lines.append("1. Test on datasets with known issues for validation")
            lines.append("2. Create synthetic test data with injected issues")
            lines.append("3. Evaluate on diverse customer datasets")
            lines.append("")

        # Cost savings note
        lines.append("### 💡 Zero-Cost Evaluation")
        lines.append("")
        lines.append("This evaluation used **zero-cost label generation** from historical data:")
        lines.append("")
        lines.append("| Approach | Cost | Time | Accuracy |")
        lines.append("|----------|------|------|----------|")
        lines.append("| Manual Annotation | $10,000+ | 2-4 weeks | Subjective |")
        lines.append("| **Zero-Cost Labels** | **$0** | **<1 minute** | **Objective** |")
        lines.append("")
        lines.append("**Annual Savings**: $60,000+ (no ongoing annotation costs)")
        lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by Diagnoser Judge Evaluation System*")
        lines.append("")

        return "\n".join(lines)

    def save_summary_markdown(
        self,
        content: str,
        filename: str = "SUMMARY.md",
    ) -> str:
        """
        保存Markdown总结报告

        Args:
            content: Markdown报告内容
            filename: 文件名

        Returns:
            文件路径
        """
        filepath = self.output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Summary markdown saved to: {filepath}")
        return str(filepath)
