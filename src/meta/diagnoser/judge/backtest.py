"""
Backtest engine for Diagnoser Judge evaluation system.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np

from .schemas import (
    BacktestResult,
    AccuracyMetrics,
)
from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    回测引擎 - 模拟实时预测场景

    核心原则:
    1. 训练数据 < 测试数据 (时间顺序)
    2. 只使用历史数据做预测
    3. 避免lookahead bias
    """

    def __init__(
        self,
        prediction_interval_days: int = 7,
        min_history_days: int = 30,
        max_history_days: int = 30,
    ):
        """
        初始化回测引擎

        Args:
            prediction_interval_days: 预测间隔 (每N天预测一次)
            min_history_days: 最少历史数据天数
            max_history_days: 最多使用历史数据天数 (默认30天滚动窗口)
        """
        self.prediction_interval_days = prediction_interval_days
        self.min_history_days = min_history_days
        self.max_history_days = max_history_days
        self.metrics_calc = MetricsCalculator()

    def run_backtest(
        self,
        detector,
        historical_data: pd.DataFrame,
        prediction_points: Optional[List[int]] = None,
        ground_truth: Optional[List[Dict[str, Any]]] = None,
    ) -> BacktestResult:
        """
        运行时间序列回测

        Args:
            detector: 检测器实例
            historical_data: 历史数据，必须按日期排序
            prediction_points: 预测时间点索引列表
            ground_truth: 真实标注 (可选)

        Returns:
            BacktestResult对象
        """
        logger.info(
            f"Running backtest: {len(historical_data)} days, "
            f"min_history={self.min_history_days}d"
        )

        # 确保数据按日期排序
        date_col = None
        for col in ["date", "date_start"]:
            if col in historical_data.columns:
                date_col = col
                historical_data = historical_data.sort_values(col)
                break

        if date_col is None and historical_data.index.name in ["date", "date_start"]:
            historical_data = historical_data.sort_index()
            date_col = historical_data.index.name

        # 生成预测点
        if prediction_points is None:
            prediction_points = self._generate_prediction_points(historical_data)

        logger.info(f"Prediction points: {len(prediction_points)}")

        # 存储预测结果
        all_predictions = []
        all_actuals = []

        # 模拟实时预测
        for pred_idx in prediction_points:
            # 只使用历史数据
            history_start = max(0, pred_idx - self.max_history_days)
            history_end = pred_idx
            history_data = historical_data.iloc[history_start:history_end].copy()

            # 检查最少数据要求
            if len(history_data) < self.min_history_days:
                logger.warning(
                    f"Skipping point {pred_idx}: insufficient history "
                    f"({len(history_data)} < {self.min_history_days})"
                )
                continue

            # 当前预测日期
            current_data = historical_data.iloc[pred_idx:pred_idx + 1].copy()

            # 运行检测器
            try:
                issues = detector.detect(history_data, "backtest")

                # 获取日期值
                date_val = None
                if date_col and date_col in current_data.columns:
                    date_val = current_data.iloc[0][date_col]
                elif len(current_data) > 0:
                    date_val = current_data.index[0]

                # 记录预测结果
                for issue in issues:
                    all_predictions.append({
                        "date": date_val,
                        "entity_id": issue.affected_entities[0] if issue.affected_entities else "",
                        "severity": issue.severity.value if hasattr(issue.severity, "value") else str(issue.severity),
                        "description": issue.description,
                        "metrics": issue.metrics,
                    })

            except Exception as e:
                logger.error(f"Error at prediction point {pred_idx}: {e}")
                continue

        # 如果有ground truth，计算准确性
        accuracy_metrics = AccuracyMetrics()
        if ground_truth is not None:
            accuracy_metrics = self.metrics_calc.calculate_accuracy(
                predictions=all_predictions,
                ground_truth=ground_truth,
            )

        result = BacktestResult(
            predictions=all_predictions,
            actuals=all_actuals,
            accuracy_metrics=accuracy_metrics,
            prediction_points=prediction_points,
            total_days=len(historical_data),
        )

        logger.info(
            f"Backtest complete: {len(all_predictions)} predictions, "
            f"F1={accuracy_metrics.f1_score:.2%}"
        )

        return result

    def _generate_prediction_points(
        self,
        data: pd.DataFrame,
    ) -> List[int]:
        """
        生成预测时间点

        从min_history_days开始，每隔prediction_interval_days预测一次

        Args:
            data: 历史数据

        Returns:
            预测点索引列表
        """
        points = []

        start_idx = self.min_history_days
        for idx in range(start_idx, len(data), self.prediction_interval_days):
            points.append(idx)

        return points

    def evaluate_stability(
        self,
        detector,
        data: pd.DataFrame,
        num_windows: int = 5,
    ) -> Dict[str, Any]:
        """
        评估检测器在不同时间窗口的稳定性

        Args:
            detector: 检测器实例
            data: 历史数据
            num_windows: 测试窗口数量

        Returns:
            稳定性分析结果
        """
        logger.info(f"Evaluating stability across {num_windows} windows")

        window_size = len(data) // num_windows
        results = []

        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, len(data))
            window_data = data.iloc[start_idx:end_idx].copy()

            if len(window_data) < self.min_history_days:
                continue

            try:
                issues = detector.detect(window_data, f"stability_test_{i}")
                results.append({
                    "window": i,
                    "num_issues": len(issues),
                    "severities": [
                        issue.severity.value if hasattr(issue.severity, "value") else str(issue.severity)
                        for issue in issues
                    ],
                })
            except Exception as e:
                logger.error(f"Error in window {i}: {e}")
                continue

        # 分析稳定性
        if not results:
            return {
                "num_windows": 0,
                "stable": False,
                "std_issues": 0,
                "mean_issues": 0,
            }

        num_issues_list = [r["num_issues"] for r in results]
        std_issues = np.std(num_issues_list)
        mean_issues = np.mean(num_issues_list)

        # 判断是否稳定 (标准差 < 均值的50%)
        stable = std_issues < (mean_issues * 0.5) if mean_issues > 0 else True

        return {
            "num_windows": len(results),
            "stable": stable,
            "std_issues": std_issues,
            "mean_issues": mean_issues,
            "cv": std_issues / (mean_issues + 1e-6),  # 变异系数
            "window_results": results,
        }

    def compare_detection_timelines(
        self,
        detector_v1,
        detector_v2,
        data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        对比两个版本的检测时间线

        Args:
            detector_v1: 版本1检测器
            detector_v2: 版本2检测器
            data: 历史数据

        Returns:
            时间线对比结果
        """
        logger.info("Comparing detection timelines")

        # 生成预测点
        prediction_points = self._generate_prediction_points(data)

        v1_timeline = []
        v2_timeline = []

        # V1检测时间线
        for pred_idx in prediction_points:
            history_data = data.iloc[:pred_idx].copy()

            if len(history_data) < self.min_history_days:
                continue

            try:
                issues = detector_v1.detect(history_data, "v1_timeline")
                v1_timeline.append({
                    "day": pred_idx,
                    "num_issues": len(issues),
                })
            except Exception as e:
                logger.error(f"V1 error at day {pred_idx}: {e}")

        # V2检测时间线
        for pred_idx in prediction_points:
            history_data = data.iloc[:pred_idx].copy()

            if len(history_data) < self.min_history_days:
                continue

            try:
                issues = detector_v2.detect(history_data, "v2_timeline")
                v2_timeline.append({
                    "day": pred_idx,
                    "num_issues": len(issues),
                })
            except Exception as e:
                logger.error(f"V2 error at day {pred_idx}: {e}")

        # 对比分析
        v1_total = sum(t["num_issues"] for t in v1_timeline)
        v2_total = sum(t["num_issues"] for t in v2_timeline)

        return {
            "v1_total_issues": v1_total,
            "v2_total_issues": v2_total,
            "v1_timeline": v1_timeline,
            "v2_timeline": v2_timeline,
            "improvement": v1_total - v2_total,
            "improvement_pct": (
                (v1_total - v2_total) / v1_total * 100
                if v1_total > 0 else 0
            ),
        }
