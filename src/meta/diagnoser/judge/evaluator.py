"""
Evaluator for Diagnoser Judge evaluation system.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np

from .schemas import (
    EvaluationResult,
    BacktestResult,
    ComparisonResult,
    AccuracyMetrics,
    ReliabilityMetrics,
    TimelinessMetrics,
    InterpretabilityMetrics,
    BusinessValueMetrics,
)
from .metrics import MetricsCalculator
from .scorer import DiagnoserScorer
from .backtest import BacktestEngine
from .label_generator import ZeroCostLabelGenerator

logger = logging.getLogger(__name__)


class DiagnoserEvaluator:
    """
    核心评估器 - 评估检测器性能

    用法:
        evaluator = DiagnoserEvaluator()

        # 评估单个检测器
        result = evaluator.evaluate(
            detector=detector,
            test_data=test_data,
            ground_truth=ground_truth
        )

        # 对比两个版本
        comparison = evaluator.compare(
            detector_v1=detector_v1,
            detector_v2=detector_v2,
            test_data=test_data
        )
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化评估器"""
        self.config = config or {}
        self.metrics_calc = MetricsCalculator()
        self.scorer = DiagnoserScorer()
        self.backtest_engine = BacktestEngine()
        self.label_generator = ZeroCostLabelGenerator()

    def evaluate(
        self,
        detector,
        test_data: pd.DataFrame,
        ground_truth: Optional[List[Dict[str, Any]]] = None,
        detector_name: str = "",
        auto_label: bool = True,
        label_method: str = "performance_drop",
    ) -> EvaluationResult:
        """
        全面评估检测器

        Args:
            detector: 检测器实例
            test_data: 测试数据
            ground_truth: 真实标注（可选，如果不提供则自动生成零成本标注）
            detector_name: 检测器名称
            auto_label: 是否自动生成零成本标注（当ground_truth为None时）
            label_method: 零成本标注方法 ("performance_drop", "rule_based", "statistical_anomaly", "combined")

        Returns:
            EvaluationResult对象
        """
        logger.info(f"Evaluating detector: {detector_name or detector.__class__.__name__}")

        result = EvaluationResult(
            detector_name=detector_name or detector.__class__.__name__,
            detector_type=getattr(detector, "__class__.__name__", ""),
            evaluation_date=datetime.now(),
        )

        # 1. 评估准确性
        if ground_truth is None and auto_label:
            # 自动生成零成本标注
            logger.info(f"Auto-generating zero-cost labels using method: {label_method}")
            ground_truth = self.label_generator.generate(
                test_data,
                method=label_method
            )
            logger.info(f"Generated {len(ground_truth)} zero-cost labels")

        if ground_truth is not None:
            result.accuracy = self._evaluate_accuracy(
                detector, test_data, ground_truth
            )
        else:
            logger.warning("No ground truth provided and auto_label=False, skipping accuracy evaluation")

        # 2. 评估可靠性
        result.reliability = self._evaluate_reliability(detector, test_data)

        # 3. 评估实时性
        result.timeliness = self._evaluate_timeliness(detector, test_data)

        # 4. 评估可解释性
        issues = detector.detect(test_data.copy(), "eval_test")
        result.interpretability = self._evaluate_interpretability(issues)

        # 5. 评估业务价值
        result.business_value = self._evaluate_business_value(issues, test_data)

        # 计算综合评分
        self.scorer.calculate_score(result)

        # 添加详细建议
        result.details["suggestions"] = self.scorer.get_suggestions(result)

        # 记录使用的标注方法
        if ground_truth is not None:
            result.details["label_method"] = label_method if auto_label else "provided"
            result.details["label_count"] = len(ground_truth)

        logger.info(f"Evaluation complete: Score={result.overall_score:.1f}/100, Grade={result.grade}")

        return result

    def _evaluate_accuracy(
        self,
        detector,
        test_data: pd.DataFrame,
        ground_truth: List[Dict[str, Any]],
    ) -> AccuracyMetrics:
        """评估准确性"""
        # 按entity分组处理
        entity_col = "ad_id" if "ad_id" in test_data.columns else None

        if entity_col is None:
            # 没有entity列，使用原始逻辑
            predictions = detector.detect(test_data.copy(), "accuracy_test")

            pred_formatted = []
            for pred in predictions:
                pred_formatted.append({
                    "affected_entities": pred.affected_entities,
                    "has_issue": True,
                })

            return self.metrics_calc.calculate_accuracy(
                predictions=pred_formatted,
                ground_truth=ground_truth,
            )

        # 按entity分组调用detector
        all_predictions = []

        entity_count = 0
        for entity_id, entity_data in test_data.groupby(entity_col):
            entity_count += 1

            # Log first few entities for debugging
            if entity_count <= 5 or str(entity_id) == '120215767837920310':
                logger.info(f"  Processing entity {entity_id} ({len(entity_data)} rows)...")

            try:
                entity_predictions = detector.detect(entity_data.copy(), entity_id)

                # Log if issues found
                if entity_predictions:
                    logger.info(f"  Entity {entity_id}: {len(entity_predictions)} issues detected")

                for pred in entity_predictions:
                    all_predictions.append({
                        "affected_entities": pred.affected_entities,
                        "has_issue": True,
                    })
            except Exception as e:
                logger.warning(f"Error detecting for entity {entity_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

        logger.info(f"  Processed {entity_count} entities, found {len(all_predictions)} issues")

        return self.metrics_calc.calculate_accuracy(
            predictions=all_predictions,
            ground_truth=ground_truth,
        )

    def _evaluate_reliability(
        self,
        detector,
        test_data: pd.DataFrame,
    ) -> ReliabilityMetrics:
        """评估可靠性"""
        return self.metrics_calc.calculate_reliability(detector, test_data)

    def _evaluate_timeliness(
        self,
        detector,
        test_data: pd.DataFrame,
    ) -> TimelinessMetrics:
        """评估实时性"""
        return self.metrics_calc.calculate_timeliness(detector, test_data)

    def _evaluate_interpretability(
        self,
        issues: List,
    ) -> InterpretabilityMetrics:
        """评估可解释性"""
        # 转换为dict格式
        issues_dict = []
        for issue in issues:
            issues_dict.append({
                "description": issue.description,
                "metrics": issue.metrics,
            })

        return self.metrics_calc.calculate_interpretability(issues_dict)

    def _evaluate_business_value(
        self,
        issues: List,
        test_data: pd.DataFrame,
    ) -> BusinessValueMetrics:
        """评估业务价值"""
        # 计算平均每日花费
        if "spend" in test_data.columns:
            daily_spend = test_data["spend"].mean()
        else:
            daily_spend = 100.0  # 默认值

        # 转换为dict格式
        issues_dict = []
        for issue in issues:
            issues_dict.append({
                "severity": issue.severity.value if hasattr(issue.severity, "value") else str(issue.severity),
                "metrics": issue.metrics,
            })

        return self.metrics_calc.calculate_business_value(issues_dict, daily_spend)

    def compare(
        self,
        detector_v1,
        detector_v2,
        test_data: pd.DataFrame,
        ground_truth: Optional[List[Dict[str, Any]]] = None,
    ) -> ComparisonResult:
        """
        对比两个检测器版本

        Args:
            detector_v1: 版本1
            detector_v2: 版本2
            test_data: 测试数据
            ground_truth: 真实标注（可选）

        Returns:
            ComparisonResult对象
        """
        logger.info("Comparing detector v1 vs v2")

        # 评估两个版本
        result_v1 = self.evaluate(detector_v1, test_data, ground_truth, "v1")
        result_v2 = self.evaluate(detector_v2, test_data, ground_truth, "v2")

        # 计算改进
        improvement = result_v2.overall_score - result_v1.overall_score
        improvement_pct = (
            (improvement / result_v1.overall_score * 100)
            if result_v1.overall_score > 0 else 0
        )

        # 判断胜负
        if improvement > 5:
            winner = "v2"
        elif improvement < -5:
            winner = "v1"
        else:
            winner = "tie"

        comparison = ComparisonResult(
            detector_v1_name=result_v1.detector_name,
            detector_v2_name=result_v2.detector_name,
            v1_score=result_v1.overall_score,
            v2_score=result_v2.overall_score,
            improvement=improvement,
            improvement_pct=improvement_pct,
            v1_metrics=result_v1.to_dict(),
            v2_metrics=result_v2.to_dict(),
            winner=winner,
        )

        logger.info(
            f"Comparison complete: "
            f"V1={comparison.v1_score:.1f}, V2={comparison.v2_score:.1f}, "
            f"Improvement={comparison.improvement:+.1f} ({comparison.improvement_pct:+.1f}%), "
            f"Winner={comparison.winner}"
        )

        return comparison

    def backtest(
        self,
        detector,
        historical_data: pd.DataFrame,
        prediction_points: Optional[List[int]] = None,
    ) -> BacktestResult:
        """
        运行时间序列回测

        Args:
            detector: 检测器实例
            historical_data: 历史数据
            prediction_points: 预测时间点列表

        Returns:
            BacktestResult对象
        """
        logger.info("Running backtest evaluation")

        return self.backtest_engine.run_backtest(
            detector=detector,
            historical_data=historical_data,
            prediction_points=prediction_points,
        )
