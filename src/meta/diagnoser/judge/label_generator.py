"""
Zero-Cost Label Generator for Diagnoser Judge.

Generates ground truth labels from historical data without manual annotation.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def _preprocess_data(data: pd.DataFrame, numeric_cols: List[str] = None) -> pd.DataFrame:
    """
    预处理数据：确保指定列是数值类型

    Args:
        data: 原始数据
        numeric_cols: 需要转换为数值的列名列表

    Returns:
        预处理后的数据
    """
    data = data.copy()

    # 处理指定的数值列
    if numeric_cols:
        for col in numeric_cols:
            if col and col in data.columns:
                # 转换为数值类型
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    return data


class ZeroCostLabelGenerator:
    """
    零成本标注生成器

    从历史数据自动生成ground truth标注，无需人工标注。

    方法：
    1. Performance Drop: 基于性能下降（ROAS/CPA变化）
    2. Rule-Based: 基于业务规则
    3. Statistical Anomaly: 基于统计异常（Z-Score, IQR）
    """

    def generate(
        self,
        data: pd.DataFrame,
        method: str = "performance_drop",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        生成零成本标注

        Args:
            data: 历史数据
            method: 标注方法
                - "performance_drop": 基于性能下降
                - "rule_based": 基于业务规则
                - "statistical_anomaly": 基于统计异常
                - "combined": 综合多种方法
            **kwargs: 方法特定的参数

        Returns:
            标注列表
        """
        logger.info(f"Generating zero-cost labels using method: {method}")

        if method == "performance_drop":
            return self._generate_performance_drop_labels(data, **kwargs)
        elif method == "rule_based":
            return self._generate_rule_based_labels(data, **kwargs)
        elif method == "statistical_anomaly":
            return self._generate_statistical_anomaly_labels(data, **kwargs)
        elif method == "combined":
            return self._generate_combined_labels(data, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _generate_performance_drop_labels(
        self,
        data: pd.DataFrame,
        entity_col: str = "ad_id",
        roas_col: str = "purchase_roas",
        cpa_col: str = None,  # 自动检测
        spend_col: str = "spend",
        window_size: int = 7,
        drop_threshold: float = 0.5,  # 50%下降
        min_spend: float = 50.0,
    ) -> List[Dict[str, Any]]:
        """
        基于性能下降生成标注

        逻辑：
        - 如果ROAS从窗口均值下降超过drop_threshold → 有问题
        - 如果CPA从窗口均值上涨超过(1/drop_threshold - 1) → 有问题

        Args:
            data: 历史数据
            entity_col: 实体ID列名
            roas_col: ROAS列名
            cpa_col: CPA列名
            spend_col: 花费列名
            window_size: 滚动窗口大小
            drop_threshold: 下降阈值（比例）
            min_spend: 最小花费（过滤低价值数据）

        Returns:
            标注列表
        """
        labels = []

        # 预处理数据：确保数值列
        numeric_cols = [roas_col, spend_col]
        if cpa_col:
            numeric_cols.append(cpa_col)
        data = _preprocess_data(data, numeric_cols)

        # 自动检测CPA列
        if cpa_col is None:
            possible_cpa_cols = [
                "cost_per_conversion",
                "cost_per_result",
                "cost_per_action_type",
            ]
            for col in possible_cpa_cols:
                if col in data.columns:
                    cpa_col = col
                    # 预处理CPA列
                    data = _preprocess_data(data, [cpa_col])
                    logger.debug(f"Auto-detected CPA column: {cpa_col}")
                    break
            if cpa_col is None:
                logger.warning("No CPA column found, skipping CPA increase detection")

        # 确保数据按日期排序
        if "date" in data.columns:
            data = data.sort_values("date")
        elif "date_start" in data.columns:
            data = data.sort_values("date_start")

        # 按实体分组
        if entity_col not in data.columns:
            logger.warning(f"Entity column '{entity_col}' not found, using single entity")
            data["_temp_entity"] = "single_entity"
            entity_col = "_temp_entity"

        for entity_id, entity_data in data.groupby(entity_col):
            entity_data = entity_data.copy().reset_index(drop=True)

            # 跳过数据不足的实体
            if len(entity_data) < window_size + 1:
                continue

            # 计算rolling metrics
            if roas_col in entity_data.columns:
                entity_data[f"rolling_{roas_col}"] = entity_data[roas_col].rolling(
                    window_size, min_periods=1
                ).mean()

            if cpa_col in entity_data.columns:
                entity_data[f"rolling_{cpa_col}"] = entity_data[cpa_col].rolling(
                    window_size, min_periods=1
                ).mean()

            # 检测performance drop
            for i in range(window_size, len(entity_data)):
                current = entity_data.iloc[i]
                before = entity_data.iloc[i - window_size:i]

                # 检查最小花费
                current_spend = current.get(spend_col, 0)
                if pd.isna(current_spend) or current_spend < min_spend:
                    continue

                # 检测ROAS下降
                if roas_col in entity_data.columns:
                    rolling_roas = current[f"rolling_{roas_col}"]
                    current_roas = current[roas_col]

                    if not pd.isna(rolling_roas) and not pd.isna(current_roas):
                        if rolling_roas > 0.1:  # 避免除零
                            drop_ratio = (rolling_roas - current_roas) / rolling_roas

                            if drop_ratio >= drop_threshold:
                                severity = "critical" if drop_ratio >= 0.7 else "high"

                                labels.append({
                                    "affected_entities": [entity_id],
                                    "has_issue": True,
                                    "issue_type": "performance_drop",
                                    "date": current.get("date", current.get("date_start")),
                                    "severity": severity,
                                    "metrics": {
                                        "before_roas": rolling_roas,
                                        "after_roas": current_roas,
                                        "drop_ratio": drop_ratio,
                                        "spend": current_spend,
                                    },
                                    "label_method": "performance_drop",
                                    "confidence": "high",
                                })

                                logger.debug(
                                    f"Label: {entity_id}, ROAS drop {drop_ratio:.1%}, "
                                    f"{rolling_roas:.2f} → {current_roas:.2f}"
                                )

                # 检测CPA上涨
                if cpa_col in entity_data.columns:
                    rolling_cpa = current[f"rolling_{cpa_col}"]
                    current_cpa = current[cpa_col]

                    if not pd.isna(rolling_cpa) and not pd.isna(current_cpa):
                        if rolling_cpa > 0:
                            increase_ratio = (current_cpa - rolling_cpa) / rolling_cpa

                            if increase_ratio >= (1 / drop_threshold - 1):  # 对应ROAS的50%下降
                                severity = "critical" if increase_ratio >= 1.0 else "high"

                                labels.append({
                                    "affected_entities": [entity_id],
                                    "has_issue": True,
                                    "issue_type": "cpa_increase",
                                    "date": current.get("date", current.get("date_start")),
                                    "severity": severity,
                                    "metrics": {
                                        "before_cpa": rolling_cpa,
                                        "after_cpa": current_cpa,
                                        "increase_ratio": increase_ratio,
                                        "spend": current_spend,
                                    },
                                    "label_method": "performance_drop",
                                    "confidence": "high",
                                })

                                logger.debug(
                                    f"Label: {entity_id}, CPA increase {increase_ratio:.1%}, "
                                    f"{rolling_cpa:.2f} → {current_cpa:.2f}"
                                )

        logger.info(f"Generated {len(labels)} performance drop labels")
        return labels

    def _generate_rule_based_labels(
        self,
        data: pd.DataFrame,
        entity_col: str = "ad_id",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        基于业务规则生成标注

        规则：
        - Creative Fatigue: 频率 > 3.0 且 CPA上涨 > 30%
        - Latency: ROAS < 1.0 持续3天
        - Dark Hours: 特定时段ROAS < 平均的50%

        Args:
            data: 历史数据
            entity_col: 实体ID列名
            **kwargs: 额外参数

        Returns:
            标注列表
        """
        labels = []

        # 预处理数据：确保数值列
        numeric_cols = ['spend', 'impressions', 'reach', 'purchase_roas']
        data = _preprocess_data(data, numeric_cols)

        # 自动检测并预处理CPA列
        cpa_col = None
        for possible_col in ["cost_per_conversion", "cost_per_result", "cost_per_action_type"]:
            if possible_col in data.columns:
                cpa_col = possible_col
                data = _preprocess_data(data, [cpa_col])
                break

        # 确保数据按日期排序
        if "date" in data.columns:
            data = data.sort_values("date")
        elif "date_start" in data.columns:
            data = data.sort_values("date_start")

        if entity_col not in data.columns:
            data["_temp_entity"] = "single_entity"
            entity_col = "_temp_entity"

        for entity_id, entity_data in data.groupby(entity_col):
            entity_data = entity_data.copy().reset_index(drop=True)

            # 跳过数据不足的实体
            if len(entity_data) < 30:
                continue

            # 规则1: Creative Fatigue
            fatigue_labels = self._apply_fatigue_rules(
                entity_data,
                entity_id,
                cpa_col,
                detector_instance=kwargs.get('detector_instance')
            )
            labels.extend(fatigue_labels)

            # 规则2: Latency
            latency_labels = self._apply_latency_rules(entity_data, entity_id)
            labels.extend(latency_labels)

            # 规则3: Dark Hours (如果有hour数据)
            if "hour" in entity_data.columns:
                dark_hours_labels = self._apply_dark_hours_rules(entity_data, entity_id)
                labels.extend(dark_hours_labels)

        logger.info(f"Generated {len(labels)} rule-based labels")
        return labels

    def _apply_fatigue_rules(
        self,
        data: pd.DataFrame,
        entity_id: str,
        cpa_col: str = None,
        detector_instance = None,
    ) -> List[Dict[str, Any]]:
        """应用疲劳检测规则

        使用与FatigueDetector相同的逻辑:
        - 使用rolling window approach (past 30 days)
        - 找黄金期 (cum_frequency between 1.0 and 2.5)
        - 检查疲劳信号
        - 需要3 consecutive days确认

        修复：使用detector的实际阈值，而非硬编码
        """
        labels = []

        # 获取detector的阈值（确保一致性）
        if detector_instance and hasattr(detector_instance, 'DEFAULT_THRESHOLDS'):
            thresholds = detector_instance.DEFAULT_THRESHOLDS
            logger.info(f"Using thresholds from detector instance for {entity_id}")
        else:
            # 后备：导入detector获取阈值
            try:
                from src.meta.diagnoser.detectors.fatigue_detector import FatigueDetector
                thresholds = FatigueDetector.DEFAULT_THRESHOLDS
                logger.info(f"Using thresholds from FatigueDetector.DEFAULT_THRESHOLDS")
            except ImportError:
                logger.warning("Cannot import FatigueDetector, using hardcoded thresholds (may be inconsistent!)")
                # 硬编码后备（与detector保持一致）
                thresholds = {
                    "window_size_days": 23,
                    "golden_min_freq": 1.0,
                    "golden_max_freq": 2.5,
                    "fatigue_freq_threshold": 3.0,
                    "cpa_increase_threshold": 1.10,  # Updated: 1.15 → 1.10
                    "consecutive_days": 1,
                    "min_golden_days": 1,
                }

        # 提取阈值参数
        window_size = thresholds["window_size_days"]
        consecutive_days = thresholds["consecutive_days"]
        min_golden_days = thresholds["min_golden_days"]
        cpa_increase_threshold = thresholds["cpa_increase_threshold"]
        golden_min_freq = thresholds["golden_min_freq"]
        golden_max_freq = thresholds["golden_max_freq"]
        fatigue_freq_threshold = thresholds["fatigue_freq_threshold"]

        logger.debug(f"Fatigue thresholds: window={window_size}, consecutive={consecutive_days}, "
                    f"min_golden={min_golden_days}, cpa_thresh={cpa_increase_threshold}")

        # Check for required columns
        required_cols = ["date_start", "spend", "impressions", "reach"]
        if not all(col in data.columns for col in required_cols):
            return labels

        # Handle conversions column
        if "conversions" not in data.columns:
            # Try to parse from actions JSON
            if "actions" in data.columns:
                data = data.copy()
                data["conversions"] = data["actions"].apply(self._parse_conversions_from_json)
            else:
                return labels

        # Sort by date
        data = data.sort_values("date_start").reset_index(drop=True)

        # Skip if insufficient data
        min_required = window_size + consecutive_days
        if len(data) < min_required:
            return labels

        detections = []

        # Start from day where we have enough data
        for i in range(min_required, len(data)):
            # ONLY use historical data (days [i-window_size : i-1])
            window = data.iloc[i - window_size : i].copy()

            # Calculate cumulative frequency within the window
            window = self._calculate_cumulative_frequency_fatigue(window)

            # Find golden period in the window
            golden_mask = (
                (window["cum_freq"] > golden_min_freq) &
                (window["cum_freq"] < golden_max_freq)
            )
            golden_period = window[golden_mask]

            if len(golden_period) < min_golden_days:
                continue

            # Calculate CPA in golden period
            total_conversions = golden_period["conversions"].sum()
            if total_conversions == 0:
                continue

            cpa_gold = golden_period["spend"].sum() / total_conversions

            # Check current day (day i) for fatigue
            current = data.iloc[i]
            current_freq = window.iloc[-1]["cum_freq"]
            current_cpa = current["spend"] / current["conversions"] if current["conversions"] > 0 else np.inf

            # Check fatigue conditions (matching FatigueDetector)
            is_fatigued = (
                current_freq > fatigue_freq_threshold and
                current_cpa > cpa_gold * cpa_increase_threshold
            )

            detections.append({
                "date": current["date_start"],
                "is_fatigued": is_fatigued,
                "current_freq": current_freq,
                "current_cpa": current_cpa,
                "cpa_gold": cpa_gold,
            })

        # Check for consecutive detections (matching FatigueDetector)
        consecutive_count = self._count_consecutive_detections_fatigue(detections)

        if consecutive_count >= consecutive_days and detections:
            # Get the most recent detection
            latest_detection = detections[-1]

            # Calculate post-fatigue metrics
            post_fatigue_start = len(data) - consecutive_count
            post_fatigue = data.iloc[post_fatigue_start:]

            current_cpa = post_fatigue["spend"].sum() / post_fatigue["conversions"].sum()
            cpa_increase_pct = ((current_cpa - latest_detection["cpa_gold"]) / latest_detection["cpa_gold"] * 100) if latest_detection["cpa_gold"] > 0 else 0

            # Determine severity
            fatigue_freq = latest_detection["current_freq"]
            if cpa_increase_pct >= 100 or fatigue_freq >= 6.0:
                severity = "critical"
            elif cpa_increase_pct >= 50 or fatigue_freq >= 4.5:
                severity = "high"
            else:
                severity = "medium"

            labels.append({
                "affected_entities": [entity_id],
                "has_issue": True,
                "issue_type": "fatigue",
                "date": data.iloc[post_fatigue_start]["date_start"],
                "severity": severity,
                "metrics": {
                    "fatigue_freq": float(fatigue_freq),
                    "cpa_gold": float(latest_detection["cpa_gold"]),
                    "current_cpa": float(current_cpa),
                    "cpa_increase_pct": float(cpa_increase_pct),
                    "post_fatigue_days": len(post_fatigue),
                    "consecutive_days": consecutive_count,
                },
                "label_method": "rule_based_fatigue",
                "confidence": "high",
            })

        return labels

    def _calculate_cumulative_frequency_fatigue(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate cumulative frequency for fatigue detection."""
        data = data.copy()
        data["cum_impressions"] = data["impressions"].cumsum()
        data["cum_reach"] = data["reach"].expanding().max()
        data["cum_freq"] = data["cum_impressions"] / data["cum_reach"].replace(0, np.nan)
        return data

    def _count_consecutive_detections_fatigue(
        self,
        detections: List[Dict[str, Any]],
    ) -> int:
        """Count consecutive fatigue detections at the end of the list."""
        if not detections:
            return 0

        consecutive_count = 0
        for detection in reversed(detections):
            if detection["is_fatigued"]:
                consecutive_count += 1
            else:
                break

        return consecutive_count

    def _parse_conversions_from_json(self, actions_str):
        """Parse conversions from actions JSON string."""
        import json
        if pd.isna(actions_str) or actions_str == "":
            return 0

        try:
            actions = json.loads(actions_str)

            purchase_keys = [
                "offsite_conversion.fb_pixel_purchase",
                "omni_purchase",
                "purchase",
                "onsite_web_purchase",
            ]

            total = 0.0
            for action in actions:
                action_type = action.get("action_type", "")
                if action_type in purchase_keys:
                    total += float(action.get("value", 0))

            return total
        except:
            return 0

    def _apply_latency_rules(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> List[Dict[str, Any]]:
        """应用延迟检测规则

        检测performance drop: ROAS从rolling平均下降超过20%
        使用与LatencyDetector相同的逻辑
        """
        labels = []

        if "purchase_roas" in data.columns and "spend" in data.columns:
            data = data.copy()

            # 确保日期排序
            date_col = "date" if "date" in data.columns else "date_start"
            data = data.sort_values(date_col).reset_index(drop=True)

            # 计算rolling平均ROAS (3天窗口)
            data["rolling_roas"] = data["purchase_roas"].rolling(3, min_periods=1).mean()

            # 检测performance drop - 使用与LatencyDetector相同的逻辑
            # 从rolling_window_days开始（通常是3）
            rolling_window_days = 3
            roas_threshold = 1.0
            min_daily_spend = 50
            min_drop_ratio = 0.2

            for i in range(rolling_window_days, len(data)):
                current = data.iloc[i]

                # 使用前一天的rolling average（与LatencyDetector保持一致）
                rolling_roas = data.loc[i - 1, "rolling_roas"]
                current_roas = current["purchase_roas"]
                current_spend = current["spend"]

                # 计算drop ratio - 只有当rolling_roas > 0时才有意义
                if rolling_roas <= 0:
                    # 没有有效的历史数据，跳过
                    continue

                drop_ratio = (rolling_roas - current_roas) / rolling_roas

                # 检查breakdown条件（与LatencyDetector相同）:
                # 1. 当前ROAS < threshold (1.0)
                # 2. Spend >= min_daily_spend (50)
                # 3. Drop ratio >= min_drop_ratio (0.2)
                is_breakdown = (
                    current_roas < roas_threshold
                    and current_spend >= min_daily_spend
                    and drop_ratio >= min_drop_ratio
                )

                if is_breakdown:
                    labels.append({
                        "affected_entities": [entity_id],
                        "has_issue": True,
                        "issue_type": "latency",
                        "date": current.get("date", current.get("date_start")),
                        "severity": "high" if drop_ratio >= 0.5 else "medium",
                        "metrics": {
                            "rolling_roas": float(rolling_roas),
                            "current_roas": float(current_roas),
                            "drop_ratio": float(drop_ratio),
                            "spend": float(current_spend),
                        },
                        "label_method": "rule_based_latency",
                        "confidence": "high",
                    })
                    break  # 只记录第一次

        return labels

    def _apply_dark_hours_rules(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> List[Dict[str, Any]]:
        """应用暗小时规则"""
        labels = []

        if "purchase_roas" in data.columns and "hour" in data.columns:
            # 计算每个小时平均ROAS
            hourly_roas = data.groupby("hour")["purchase_roas"].mean()
            overall_avg = hourly_roas.mean()

            # 找到表现差的小时
            underperforming_hours = hourly_roas[hourly_roas < overall_avg * 0.5].index

            for hour in underperforming_hours:
                hour_data = data[data["hour"] == hour]
                if len(hour_data) >= 3:  # 至少3天数据
                    labels.append({
                        "affected_entities": [entity_id],
                        "has_issue": True,
                        "issue_type": "dark_hours",
                        "date": hour_data.iloc[0].get("date", hour_data.iloc[0].get("date_start")),
                        "severity": "medium",
                        "metrics": {
                            "hour": int(hour),
                            "hour_roas": hourly_roas[hour],
                            "avg_roas": overall_avg,
                            "underperform_ratio": hourly_roas[hour] / overall_avg,
                        },
                        "label_method": "rule_based_dark_hours",
                        "confidence": "medium",
                    })

        return labels

    def _generate_statistical_anomaly_labels(
        self,
        data: pd.DataFrame,
        entity_col: str = "ad_id",
        metric_col: str = "purchase_roas",
        method: str = "zscore",
        threshold: float = 3.0,
    ) -> List[Dict[str, Any]]:
        """
        基于统计异常生成标注

        Args:
            data: 历史数据
            entity_col: 实体ID列名
            metric_col: 指标列名
            method: 统计方法 ("zscore", "iqr")
            threshold: 异常阈值

        Returns:
            标注列表
        """
        labels = []

        # 预处理数据：确保metric列是数值类型
        data = _preprocess_data(data, [metric_col])

        if metric_col not in data.columns:
            logger.warning(f"Metric column '{metric_col}' not found")
            return labels

        # 确保数据按日期排序
        if "date" in data.columns:
            data = data.sort_values("date")
        elif "date_start" in data.columns:
            data = data.sort_values("date_start")

        if entity_col not in data.columns:
            data["_temp_entity"] = "single_entity"
            entity_col = "_temp_entity"

        for entity_id, entity_data in data.groupby(entity_col):
            entity_data = entity_data.copy().reset_index(drop=True)

            # 移除NaN值
            metric_values = entity_data[metric_col].dropna()

            if len(metric_values) < 10:
                continue

            # Z-Score方法
            if method == "zscore":
                z_scores = np.abs(stats.zscore(metric_values))

                for idx, z_score in enumerate(z_scores):
                    if z_score > threshold:
                        original_idx = metric_values.index[idx]
                        labels.append({
                            "affected_entities": [entity_id],
                            "has_issue": True,
                            "issue_type": "statistical_anomaly",
                            "date": entity_data.iloc[original_idx].get("date", entity_data.iloc[original_idx].get("date_start")),
                            "severity": "high" if z_score > 4 else "medium",
                            "metrics": {
                                "metric_name": metric_col,
                                "metric_value": float(metric_values.iloc[idx]),
                                "z_score": float(z_score),
                                "method": "zscore",
                            },
                            "label_method": "statistical_anomaly_zscore",
                            "confidence": "high",
                        })

            # IQR方法
            elif method == "iqr":
                Q1 = metric_values.quantile(0.25)
                Q3 = metric_values.quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                anomalies = metric_values[(metric_values < lower_bound) | (metric_values > upper_bound)]

                for idx, value in anomalies.items():
                    labels.append({
                        "affected_entities": [entity_id],
                        "has_issue": True,
                        "issue_type": "statistical_anomaly",
                        "date": entity_data.iloc[idx].get("date", entity_data.iloc[idx].get("date_start")),
                        "severity": "high",
                        "metrics": {
                            "metric_name": metric_col,
                            "metric_value": float(value),
                            "lower_bound": float(lower_bound),
                            "upper_bound": float(upper_bound),
                            "method": "iqr",
                        },
                        "label_method": "statistical_anomaly_iqr",
                        "confidence": "high",
                    })

        logger.info(f"Generated {len(labels)} statistical anomaly labels (method={method})")
        return labels

    def _generate_combined_labels(
        self,
        data: pd.DataFrame,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        综合多种方法生成标注

        合并去重，取并集
        """
        all_labels = []

        # 方法1: Performance Drop
        perf_labels = self._generate_performance_drop_labels(data, **kwargs)
        all_labels.extend(perf_labels)

        # 方法2: Rule-Based
        rule_labels = self._generate_rule_based_labels(data, **kwargs)
        all_labels.extend(rule_labels)

        # 去重（基于entity + date + issue_type）
        unique_labels = []
        seen = set()

        for label in all_labels:
            entity = label["affected_entities"][0] if label["affected_entities"] else None
            date = label.get("date")
            issue_type = label["issue_type"]

            key = (entity, date, issue_type)

            if key not in seen:
                seen.add(key)
                unique_labels.append(label)

        logger.info(f"Generated {len(unique_labels)} combined labels (raw={len(all_labels)})")
        return unique_labels
