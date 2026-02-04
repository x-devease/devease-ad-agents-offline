"""
Diagnoser Judge - 评估系统

用于衡量 Diagnoser 是否完成目标，评估检测器性能和质量。

核心特性：
- 零成本标注生成（无需人工标注）
- 5个评估维度
- 30天滚动窗口回测
- JSON报告输出
"""

from .evaluator import DiagnoserEvaluator
from .metrics import (
    AccuracyMetrics,
    ReliabilityMetrics,
    TimelinessMetrics,
    InterpretabilityMetrics,
    BusinessValueMetrics,
    EvaluationResult,
)
from .scorer import DiagnoserScorer
from .reporter import EvaluationReporter
from .backtest import BacktestEngine
from .label_generator import ZeroCostLabelGenerator

__all__ = [
    "DiagnoserEvaluator",
    "AccuracyMetrics",
    "ReliabilityMetrics",
    "TimelinessMetrics",
    "InterpretabilityMetrics",
    "BusinessValueMetrics",
    "EvaluationResult",
    "DiagnoserScorer",
    "EvaluationReporter",
    "BacktestEngine",
    "ZeroCostLabelGenerator",
]
