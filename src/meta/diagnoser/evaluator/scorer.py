"""
Scorer for Diagnoser Judge evaluation system.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List

from .schemas import (
    AccuracyMetrics,
    ReliabilityMetrics,
    TimelinessMetrics,
    InterpretabilityMetrics,
    BusinessValueMetrics,
    EvaluationResult,
)

logger = logging.getLogger(__name__)


def _reliability_to_score(metrics: ReliabilityMetrics) -> float:
    """将可靠性指标转换为评分 (0-100)"""
    # 加权平均
    score = (
        metrics.stability_score * 0.4 +
        metrics.reproducibility_score * 0.3 +
        metrics.robustness_score * 0.3
    ) * 100
    return min(100, max(0, score))


def _timeliness_to_score(metrics: TimelinessMetrics) -> float:
    """将实时性指标转换为评分 (0-100)"""
    score = 100

    # 检测延迟扣分（每延迟1天扣5分）
    score -= min(40, metrics.detection_delay_days * 5)

    # 处理时间扣分（每100ms扣1分）
    score -= min(20, metrics.processing_time_ms / 100)

    # 数据需求扣分（每超过7天扣2分）
    excess_days = max(0, metrics.min_data_days - 7)
    score -= min(40, excess_days * 2)

    return max(0, score)


def _interpretability_to_score(metrics: InterpretabilityMetrics) -> float:
    """将可解释性指标转换为评分 (0-100)"""
    # 加权平均
    score = (
        metrics.transparency_score * 0.4 +
        metrics.readability_score * 0.3 +
        metrics.actionability_score * 0.3
    ) * 100
    return min(100, max(0, score))


def _business_value_to_score(metrics: BusinessValueMetrics) -> float:
    """将业务价值指标转换为评分 (0-100)"""
    score = 100

    # 节省金额评分（每$1000得1分，最高50分）
    savings_score = min(50, metrics.estimated_savings_usd / 1000)
    score = savings_score

    # 改进空间评分
    score += metrics.improvement_potential * 30

    # 用户满意度
    score += metrics.user_satisfaction * 20

    return min(100, max(0, score))


class DiagnoserScorer:
    """
    评分系统 - 计算综合评分和等级

    评分权重:
    - 准确性: 40%
    - 可靠性: 20%
    - 实时性: 20%
    - 可解释性: 10%
    - 业务价值: 10%
    """

    # 权重配置
    WEIGHTS = {
        "accuracy": 0.40,
        "reliability": 0.20,
        "timeliness": 0.20,
        "interpretability": 0.10,
        "business_value": 0.10,
    }

    # 评分转换函数
    SCORE_FUNCTIONS = {
        "accuracy": lambda m: m.f1_score * 100,
        "reliability": _reliability_to_score,
        "timeliness": _timeliness_to_score,
        "interpretability": _interpretability_to_score,
        "business_value": _business_value_to_score,
    }

    def calculate_score(self, result: EvaluationResult) -> float:
        """
        计算综合评分 (0-100)

        Args:
            result: 评估结果

        Returns:
            综合评分
        """
        scores = {}

        # 计算各维度评分
        scores["accuracy"] = self.SCORE_FUNCTIONS["accuracy"](result.accuracy)
        scores["reliability"] = self.SCORE_FUNCTIONS["reliability"](result.reliability)
        scores["timeliness"] = self.SCORE_FUNCTIONS["timeliness"](result.timeliness)
        scores["interpretability"] = self.SCORE_FUNCTIONS["interpretability"](result.interpretability)
        scores["business_value"] = self.SCORE_FUNCTIONS["business_value"](result.business_value)

        # 加权求和
        overall_score = sum(
            scores[dim] * weight
            for dim, weight in self.WEIGHTS.items()
        )

        result.overall_score = overall_score
        result.grade = self.get_grade(overall_score)

        return overall_score

    def get_grade(self, score: float) -> str:
        """
        将评分转换为等级

        Args:
            score: 综合评分 (0-100)

        Returns:
            等级 (A+, A, B, C, F)
        """
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        else:
            return "F"

    def get_suggestions(self, result: EvaluationResult) -> List[str]:
        """
        根据评估结果生成改进建议

        Args:
            result: 评估结果

        Returns:
            建议列表
        """
        suggestions = []

        # 准确性建议
        if result.accuracy.f1_score < 0.7:
            if result.accuracy.precision < 0.7:
                suggestions.append(
                    "降低误报率：考虑提高检测阈值或增加确认条件"
                )
            if result.accuracy.recall < 0.7:
                suggestions.append(
                    "提高召回率：考虑降低检测阈值或增加敏感度"
                )

        # 实时性建议
        if result.timeliness.processing_time_ms > 1000:
            suggestions.append(
                f"优化处理速度：当前耗时 {result.timeliness.processing_time_ms:.0f}ms，建议优化算法或使用缓存"
            )

        if result.timeliness.min_data_days > 30:
            suggestions.append(
                f"减少数据需求：当前需要 {result.timeliness.min_data_days} 天，考虑使用增量检测"
            )

        # 可解释性建议
        if result.interpretability.transparency_score < 0.7:
            suggestions.append(
                "提高透明度：在结果中添加更多算法说明和推理过程"
            )

        if result.interpretability.actionability_score < 0.7:
            suggestions.append(
                "增强可操作性：提供更具体的优化建议和行动步骤"
            )

        # 业务价值建议
        if result.business_value.estimated_savings_usd < 1000:
            suggestions.append(
                "提升业务价值：检测到的问题影响较小，考虑调整检测目标"
            )

        return suggestions
