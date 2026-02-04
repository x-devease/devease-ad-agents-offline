"""
Metrics calculation for Diagnoser Judge evaluation system.
"""

from __future__ import annotations

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from .schemas import (
    AccuracyMetrics,
    ReliabilityMetrics,
    TimelinessMetrics,
    InterpretabilityMetrics,
    BusinessValueMetrics,
    EvaluationResult,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """计算各种评估指标"""

    @staticmethod
    def calculate_accuracy(
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        entity_match_key: str = "entity_id",
    ) -> AccuracyMetrics:
        """
        计算准确性指标

        Args:
            predictions: 检测器预测的问题列表
            ground_truth: 实际标注的问题列表
            entity_match_key: 用于匹配的key

        Returns:
            AccuracyMetrics对象
        """
        # ========== 鲁棒性检查 ==========
        # 1. 处理空输入
        if not predictions:
            predictions = []
        if not ground_truth:
            ground_truth = []

        # 2. 确保是列表类型
        if not isinstance(predictions, list):
            logger.warning(f"predictions不是列表类型: {type(predictions)}, 转换为空列表")
            predictions = []
        if not isinstance(ground_truth, list):
            logger.warning(f"ground_truth不是列表类型: {type(ground_truth)}, 转换为空列表")
            ground_truth = []

        # ========== 辅助函数：安全提取实体列表 ==========
        def _safe_extract_entities(item: Dict[str, Any], key: str = "affected_entities") -> list:
            """安全地提取实体列表，处理各种边界情况"""
            if not isinstance(item, dict):
                return []

            entities = item.get(key, [])

            # 处理None
            if entities is None:
                return []

            # 如果是单个字符串，转换为列表
            if isinstance(entities, str):
                if entities:  # 非空字符串
                    return [entities]
                return []

            # 确保是列表
            if not isinstance(entities, list):
                logger.warning(f"entities类型错误: {type(entities)}, 期望list")
                return []

            # 过滤掉None和空字符串
            return [
                str(e) for e in entities
                if e is not None and str(e).strip() != ""
            ]

        # ========== 提取预测实体 ==========
        pred_entities = set()
        pred_entities_with_issue = set()

        for pred in predictions:
            entities = _safe_extract_entities(pred, "affected_entities")
            for entity in entities:
                pred_entities.add(entity)
                pred_entities_with_issue.add(entity)

        # ========== 提取Ground Truth实体 ==========
        gt_entities = set()
        gt_entities_with_issue = set()

        for gt in ground_truth:
            entities = _safe_extract_entities(gt, "affected_entities")
            for entity in entities:
                gt_entities.add(entity)

                # 检查是否有问题（默认True，因为ground truth通常表示有问题）
                has_issue = gt.get("has_issue", True)
                if isinstance(has_issue, bool) and has_issue:
                    gt_entities_with_issue.add(entity)
                elif has_issue in [1, "1", "yes", "true"]:
                    gt_entities_with_issue.add(entity)

        # ========== 计算TP, FP, TN, FN ==========
        tp = len(pred_entities_with_issue & gt_entities_with_issue)
        fp = len(pred_entities_with_issue - gt_entities_with_issue)
        fn = len(gt_entities_with_issue - pred_entities_with_issue)

        # Debug logging
        if fn > 0 or fp > 0:
            logger.warning(f"  Pred entities with issue ({len(pred_entities_with_issue)}): {pred_entities_with_issue}")
            logger.warning(f"  GT entities with issue ({len(gt_entities_with_issue)}): {gt_entities_with_issue}")
            logger.warning(f"  FN entities ({fn}): {gt_entities_with_issue - pred_entities_with_issue}")
            logger.warning(f"  FP entities ({fp}): {pred_entities_with_issue - gt_entities_with_issue}")

        # TN: 所有GT实体中没有问题的，且没有被预测为有问题的
        gt_entities_no_issue = gt_entities - gt_entities_with_issue
        tn = len(gt_entities_no_issue - pred_entities_with_issue)

        # ========== 计算Precision, Recall, F1 ==========
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        # ========== 日志记录 ==========
        logger.debug(
            f"Accuracy计算: TP={tp}, FP={fp}, TN={tn}, FN={fn}, "
            f"Precision={precision:.2%}, Recall={recall:.2%}, F1={f1:.2%}"
        )

        return AccuracyMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
        )

    @staticmethod
    def calculate_reliability(
        detector,
        test_data: pd.DataFrame,
        num_runs: int = 3,
    ) -> ReliabilityMetrics:
        """
        计算可靠性指标

        Args:
            detector: 检测器实例
            test_data: 测试数据
            num_runs: 运行次数

        Returns:
            ReliabilityMetrics对象
        """
        results = []

        # 多次运行检测
        for i in range(num_runs):
            start_time = time.time()
            issues = detector.detect(test_data.copy(), f"test_{i}")
            elapsed = time.time() - start_time

            # 记录检测到的问题数量
            results.append({
                "num_issues": len(issues),
                "severity_counts": _count_severities(issues),
            })

        # 计算稳定性（标准差）
        num_issues_list = [r["num_issues"] for r in results]
        stability_score = 1.0 - (np.std(num_issues_list) / (np.mean(num_issues_list) + 1e-6))
        stability_score = max(0, min(1, stability_score))

        # 计算可重复性（完全一致的比例）
        if len(num_issues_list) > 1:
            all_same = all(x == num_issues_list[0] for x in num_issues_list)
            reproducibility_score = 1.0 if all_same else 0.5
        else:
            reproducibility_score = 1.0

        # 计算鲁棒性（对噪声的敏感度）
        # 这里简化处理：通过多次运行的一致性来推断
        robustness_score = (stability_score + reproducibility_score) / 2

        return ReliabilityMetrics(
            stability_score=stability_score,
            reproducibility_score=reproducibility_score,
            robustness_score=robustness_score,
        )

    @staticmethod
    def calculate_timeliness(
        detector,
        test_data: pd.DataFrame,
    ) -> TimelinessMetrics:
        """
        计算实时性指标

        Args:
            detector: 检测器实例
            test_data: 测试数据

        Returns:
            TimelinessMetrics对象
        """
        # 测量处理时间
        start_time = time.time()
        issues = detector.detect(test_data.copy(), "timeliness_test")
        processing_time_ms = (time.time() - start_time) * 1000

        # 获取最少数据天数
        if hasattr(detector, "thresholds"):
            if "window_size_days" in detector.thresholds:
                window_size = detector.thresholds["window_size_days"]
                consecutive_days = detector.thresholds.get("consecutive_days", 0)
                min_data_days = window_size + consecutive_days
            else:
                min_data_days = 7  # 默认值
        else:
            min_data_days = 7

        # 估算检测延迟（简化：假设问题在第X天出现，我们在第X+Y天检测到）
        # 这里用min_data_days作为代理指标
        detection_delay_days = float(min_data_days) * 0.5  # 假设平均一半窗口大小后检测到

        return TimelinessMetrics(
            detection_delay_days=detection_delay_days,
            processing_time_ms=processing_time_ms,
            min_data_days=min_data_days,
        )

    @staticmethod
    def calculate_interpretability(
        issues: List[Dict[str, Any]],
    ) -> InterpretabilityMetrics:
        """
        计算可解释性指标

        Args:
            issues: 检测到的问题列表

        Returns:
            InterpretabilityMetrics对象
        """
        if not issues:
            return InterpretabilityMetrics()

        # 透明度：是否有清晰的描述
        has_description = all(
            issue.get("description", "") != "" for issue in issues
        )
        transparency_score = 1.0 if has_description else 0.0

        # 可读性：描述的详细程度
        avg_desc_length = np.mean([
            len(issue.get("description", "")) for issue in issues
        ])
        readability_score = min(1.0, avg_desc_length / 100)  # 100字符为满分

        # 可操作性：是否有metrics和建议
        has_metrics = all(
            "metrics" in issue or len(issue.get("metrics", {})) > 0
            for issue in issues
        )
        actionability_score = 1.0 if has_metrics else 0.5

        return InterpretabilityMetrics(
            transparency_score=transparency_score,
            readability_score=readability_score,
            actionability_score=actionability_score,
        )

    @staticmethod
    def calculate_business_value(
        issues: List[Dict[str, Any]],
        daily_spend: float = 100.0,
    ) -> BusinessValueMetrics:
        """
        计算业务价值指标

        Args:
            issues: 检测到的问题列表
            daily_spend: 平均每日花费

        Returns:
            BusinessValueMetrics对象
        """
        if not issues:
            return BusinessValueMetrics()

        total_savings = 0.0

        for issue in issues:
            severity = issue.get("severity", "low")
            metrics = issue.get("metrics", {})

            # 根据严重度估算节省比例
            severity_multiplier = {
                "critical": 0.3,  # 可节省30%
                "high": 0.2,  # 可节省20%
                "medium": 0.1,  # 可节省10%
                "low": 0.05,  # 可节省5%
            }

            # 估算受影响的天数
            if "post_fatigue_days" in metrics:
                days = metrics["post_fatigue_days"]
            elif "bleeding_days" in metrics:
                days = metrics["bleeding_days"]
            else:
                days = 30  # 默认30天

            # 计算节省金额
            savings = daily_spend * days * severity_multiplier.get(severity, 0.1)
            total_savings += savings

        # 改进空间：基于问题数量和严重度
        high_severity_count = sum(
            1 for i in issues if i.get("severity") in ["critical", "high"]
        )
        improvement_potential = min(1.0, high_severity_count / 10)

        # 用户满意度（简化：基于问题质量）
        has_actionable_insights = any(
            len(i.get("metrics", {})) > 2 for i in issues
        )
        user_satisfaction = 0.7 if has_actionable_insights else 0.5

        return BusinessValueMetrics(
            estimated_savings_usd=total_savings,
            improvement_potential=improvement_potential,
            user_satisfaction=user_satisfaction,
        )


def _count_severities(issues: List) -> Dict[str, int]:
    """统计严重度分布"""
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    for issue in issues:
        severity = getattr(issue, "severity", None)
        if severity:
            severity_value = severity.value if hasattr(severity, "value") else str(severity)
            counts[severity_value] = counts.get(severity_value, 0) + 1

    return counts
