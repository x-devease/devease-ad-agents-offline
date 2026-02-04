"""
Data models for Diagnoser Judge evaluation system.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np


class DetectorType(Enum):
    """Detector types"""
    FATIGUE = "fatigue"
    LATENCY = "latency"
    DARK_HOURS = "dark_hours"


@dataclass
class AccuracyMetrics:
    """准确性指标"""
    precision: float = 0.0  # TP / (TP + FP)
    recall: float = 0.0  # TP / (TP + FN)
    f1_score: float = 0.0  # Harmonic mean
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    confusion_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class ReliabilityMetrics:
    """可靠性指标"""
    stability_score: float = 0.0  # 相似数据产生相似结果
    reproducibility_score: float = 0.0  # 多次运行结果一致
    robustness_score: float = 0.0  # 对噪声不敏感

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stability_score": self.stability_score,
            "reproducibility_score": self.reproducibility_score,
            "robustness_score": self.robustness_score,
        }


@dataclass
class TimelinessMetrics:
    """实时性指标"""
    detection_delay_days: float = 0.0  # 从问题出现到检测到的时间
    processing_time_ms: float = 0.0  # 算法运行耗时
    min_data_days: int = 0  # 最少需要多少天数据

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_delay_days": self.detection_delay_days,
            "processing_time_ms": self.processing_time_ms,
            "min_data_days": self.min_data_days,
        }


@dataclass
class InterpretabilityMetrics:
    """可解释性指标"""
    transparency_score: float = 0.0  # 算法逻辑清晰度
    readability_score: float = 0.0  # 输出结果易懂度
    actionability_score: float = 0.0  # 建议可操作性

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transparency_score": self.transparency_score,
            "readability_score": self.readability_score,
            "actionability_score": self.actionability_score,
        }


@dataclass
class BusinessValueMetrics:
    """业务价值指标"""
    estimated_savings_usd: float = 0.0  # 估算节省金额
    improvement_potential: float = 0.0  # 改进空间
    user_satisfaction: float = 0.0  # 用户满意度

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimated_savings_usd": self.estimated_savings_usd,
            "improvement_potential": self.improvement_potential,
            "user_satisfaction": self.user_satisfaction,
        }


@dataclass
class EvaluationResult:
    """评估结果"""
    detector_name: str = ""
    detector_type: str = ""
    evaluation_date: datetime = None

    # 各维度指标
    accuracy: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    reliability: ReliabilityMetrics = field(default_factory=ReliabilityMetrics)
    timeliness: TimelinessMetrics = field(default_factory=TimelinessMetrics)
    interpretability: InterpretabilityMetrics = field(default_factory=InterpretabilityMetrics)
    business_value: BusinessValueMetrics = field(default_factory=BusinessValueMetrics)

    # 综合评分
    overall_score: float = 0.0
    grade: str = "F"

    # 详细信息
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "detector_type": self.detector_type,
            "evaluation_date": self.evaluation_date.isoformat() if self.evaluation_date else None,
            "accuracy": self.accuracy.to_dict(),
            "reliability": self.reliability.to_dict(),
            "timeliness": self.timeliness.to_dict(),
            "interpretability": self.interpretability.to_dict(),
            "business_value": self.business_value.to_dict(),
            "overall_score": self.overall_score,
            "grade": self.grade,
            "details": self.details,
        }


@dataclass
class BacktestResult:
    """回测结果"""
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    actuals: List[Dict[str, Any]] = field(default_factory=list)
    accuracy_metrics: AccuracyMetrics = field(default_factory=AccuracyMetrics)

    prediction_points: List[int] = field(default_factory=list)
    total_days: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy_metrics": self.accuracy_metrics.to_dict(),
            "prediction_points": self.prediction_points,
            "total_days": self.total_days,
            "num_predictions": len(self.predictions),
        }


@dataclass
class ComparisonResult:
    """对比结果"""
    detector_v1_name: str = ""
    detector_v2_name: str = ""
    v1_score: float = 0.0
    v2_score: float = 0.0
    improvement: float = 0.0
    improvement_pct: float = 0.0

    v1_metrics: Dict[str, Any] = field(default_factory=dict)
    v2_metrics: Dict[str, Any] = field(default_factory=dict)

    winner: str = ""  # "v1", "v2", or "tie"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector_v1_name": self.detector_v1_name,
            "detector_v2_name": self.detector_v2_name,
            "v1_score": self.v1_score,
            "v2_score": self.v2_score,
            "improvement": self.improvement,
            "improvement_pct": self.improvement_pct,
            "v1_metrics": self.v1_metrics,
            "v2_metrics": self.v2_metrics,
            "winner": self.winner,
        }
