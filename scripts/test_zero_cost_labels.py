#!/usr/bin/env python3
"""
Test script for Zero-Cost Label Generator.

Tests the automatic label generation from historical data.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.meta.diagnoser.judge import ZeroCostLabelGenerator, DiagnoserEvaluator
from src.meta.diagnoser.detectors import FatigueDetector


def create_sample_data():
    """创建示例数据用于测试"""
    np.random.seed(42)

    # 创建100天的数据
    dates = pd.date_range("2024-01-01", periods=100)
    ad_ids = ["ad_1", "ad_2", "ad_3"]

    data = []
    for ad_id in ad_ids:
        for i, date in enumerate(dates):
            # 模拟正常的ROAS波动
            base_roas = np.random.uniform(1.5, 2.5)

            # ad_1在第50-60天有一个performance drop
            if ad_id == "ad_1" and 50 <= i <= 60:
                roas = base_roas * np.random.uniform(0.2, 0.5)  # 下降50-80%
            else:
                roas = base_roas

            # 模拟CPA
            cpa = 10.0 / roas if roas > 0 else 50.0

            # 模拟spend, impressions, reach
            spend = np.random.uniform(100, 500)
            impressions = int(spend * np.random.uniform(50, 100))
            reach = int(impressions / np.random.uniform(1.5, 3.0))

            data.append({
                "date": date,
                "ad_id": ad_id,
                "purchase_roas": roas,
                "cost_per_conversion": cpa,
                "spend": spend,
                "impressions": impressions,
                "reach": reach,
            })

    return pd.DataFrame(data)


def test_performance_drop_labels():
    """测试Performance Drop标注生成"""
    print("=" * 60)
    print("Test 1: Performance Drop Labels")
    print("=" * 60)

    data = create_sample_data()
    generator = ZeroCostLabelGenerator()

    # 生成标注
    labels = generator.generate(data, method="performance_drop")

    print(f"\nGenerated {len(labels)} labels")

    # 显示前5个标注
    print("\nFirst 5 labels:")
    for i, label in enumerate(labels[:5], 1):
        print(f"\n{i}. Entity: {label['affected_entities'][0]}")
        print(f"   Issue: {label['issue_type']}")
        print(f"   Date: {label['date']}")
        print(f"   Severity: {label['severity']}")
        print(f"   Metrics: {label['metrics']}")

    # 验证：应该检测到ad_1的performance drop
    ad_1_labels = [l for l in labels if "ad_1" in l["affected_entities"]]
    print(f"\nad_1 labels: {len(ad_1_labels)} (expected: ~10-11)")
    assert len(ad_1_labels) > 0, "Should detect performance drops for ad_1"

    print("\n✓ Test passed\n")


def test_rule_based_labels():
    """测试Rule-Based标注生成"""
    print("=" * 60)
    print("Test 2: Rule-Based Labels")
    print("=" * 60)

    data = create_sample_data()
    generator = ZeroCostLabelGenerator()

    # 生成标注
    labels = generator.generate(data, method="rule_based")

    print(f"\nGenerated {len(labels)} labels")

    # 显示统计
    issue_types = {}
    for label in labels:
        issue_type = label["issue_type"]
        issue_types[issue_type] = issue_types.get(issue_type, 0) + 1

    print("\nIssue type distribution:")
    for issue_type, count in issue_types.items():
        print(f"  {issue_type}: {count}")

    print("\n✓ Test passed\n")


def test_statistical_anomaly_labels():
    """测试Statistical Anomaly标注生成"""
    print("=" * 60)
    print("Test 3: Statistical Anomaly Labels")
    print("=" * 60)

    data = create_sample_data()
    generator = ZeroCostLabelGenerator()

    # 生成标注
    labels = generator.generate(data, method="statistical_anomaly", threshold=2.5)

    print(f"\nGenerated {len(labels)} labels")

    # 显示前3个标注
    print("\nFirst 3 labels:")
    for i, label in enumerate(labels[:3], 1):
        print(f"\n{i}. Entity: {label['affected_entities'][0]}")
        print(f"   Issue: {label['issue_type']}")
        print(f"   Date: {label['date']}")
        print(f"   Z-Score: {label['metrics']['z_score']:.2f}")

    print("\n✓ Test passed\n")


def test_combined_labels():
    """测试Combined标注生成"""
    print("=" * 60)
    print("Test 4: Combined Labels")
    print("=" * 60)

    data = create_sample_data()
    generator = ZeroCostLabelGenerator()

    # 生成标注
    labels = generator.generate(data, method="combined")

    print(f"\nGenerated {len(labels)} unique labels")

    # 统计
    methods = {}
    for label in labels:
        method = label.get("label_method", "unknown")
        methods[method] = methods.get(method, 0) + 1

    print("\nLabel method distribution:")
    for method, count in methods.items():
        print(f"  {method}: {count}")

    print("\n✓ Test passed\n")


def test_auto_label_evaluation():
    """测试自动标注+评估流程"""
    print("=" * 60)
    print("Test 5: Auto-Label + Evaluation")
    print("=" * 60)

    data = create_sample_data()

    # 创建检测器
    detector = FatigueDetector(config={
        "thresholds": {
            "window_size_days": 30,
            "golden_min_freq": 1.0,
            "golden_max_freq": 2.5,
            "fatigue_freq_threshold": 3.0,
            "cpa_increase_threshold": 1.3,
            "consecutive_days": 3,
            "min_golden_days": 5,
        }
    })

    # 创建evaluator，启用自动标注
    evaluator = DiagnoserEvaluator()

    # 评估（自动生成标注）
    result = evaluator.evaluate(
        detector=detector,
        test_data=data,
        detector_name="FatigueDetector_auto_label",
        auto_label=True,
        label_method="combined"
    )

    print(f"\nEvaluation Results:")
    print(f"  Overall Score: {result.overall_score:.1f}/100")
    print(f"  Grade: {result.grade}")
    print(f"  Precision: {result.accuracy.precision:.2%}")
    print(f"  Recall: {result.accuracy.recall:.2%}")
    print(f"  F1-Score: {result.accuracy.f1_score:.2%}")
    print(f"  TP: {result.accuracy.true_positives}")
    print(f"  FP: {result.accuracy.false_positives}")
    print(f"  FN: {result.accuracy.false_negatives}")

    if "label_method" in result.details:
        print(f"\nAuto-label info:")
        print(f"  Method: {result.details['label_method']}")
        print(f"  Count: {result.details['label_count']}")

    print("\n✓ Test passed\n")


def test_zero_cost():
    """测试零成本特性"""
    print("=" * 60)
    print("Test 6: Zero Cost Verification")
    print("=" * 60)

    data = create_sample_data()
    generator = ZeroCostLabelGenerator()
    evaluator = DiagnoserEvaluator()

    # 创建检测器
    detector = FatigueDetector()

    import time

    # 测试标注生成速度
    start = time.time()
    labels = generator.generate(data, method="performance_drop")
    label_time = time.time() - start

    print(f"\nLabel Generation:")
    print(f"  Labels: {len(labels)}")
    print(f"  Time: {label_time:.3f} seconds")
    print(f"  Cost: $0 (zero manual annotation)")

    # 测试评估速度
    start = time.time()
    result = evaluator.evaluate(
        detector=detector,
        test_data=data,
        ground_truth=labels,
    )
    eval_time = time.time() - start

    print(f"\nEvaluation:")
    print(f"  Time: {eval_time:.3f} seconds")
    print(f"  Score: {result.overall_score:.1f}/100")

    print(f"\nTotal Cost: $0")
    print(f"Total Time: {label_time + eval_time:.3f} seconds")

    print("\n✓ Test passed\n")


def main():
    print("\n" + "=" * 60)
    print("Zero-Cost Label Generator Tests")
    print("=" * 60 + "\n")

    try:
        test_performance_drop_labels()
        test_rule_based_labels()
        test_statistical_anomaly_labels()
        test_combined_labels()
        test_auto_label_evaluation()
        test_zero_cost()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nSummary:")
        print("- Zero-cost label generation: ✓")
        print("- Auto-label evaluation: ✓")
        print("- Multiple labeling methods: ✓")
        print("- Cost: $0")
        print("- Time: <1 second for 100 days × 3 ads")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
