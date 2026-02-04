# Diagnoser Scripts Guide

This directory contains consolidated scripts for detector evaluation and optimization. All scripts use shared utility modules to eliminate code duplication.

## Directory Structure

```
scripts/
├── README.md                          # This file
├── eval/                              # Evaluation scripts
│   ├── eval_detector.py              # Universal detector evaluation
│   └── check_detector_satisfaction.py # Check if detectors meet targets
├── optimize/                          # Optimization scripts
│   └── optimize_detectors.py         # Unified detector optimization
└── utils/                             # Shared utility modules
    ├── data_loader.py                # Data loading utilities
    ├── sliding_windows.py            # Window generation
    ├── evaluation_runner.py          # Evaluation execution
    ├── results_aggregator.py         # Results aggregation
    └── metrics_utils.py              # Metrics checking
```

## Evaluation Scripts

### eval_detector.py

Universal detector evaluation script using sliding window backtesting.

**Features:**
- Supports all detectors (FatigueDetector, LatencyDetector, DarkHoursDetector)
- Configurable window parameters
- Automatic report generation
- CLI-based configuration

**Usage:**

```bash
# Evaluate FatigueDetector with defaults
python eval/eval_detector.py --detector FatigueDetector

# Evaluate LatencyDetector with custom windows
python eval/eval_detector.py --detector LatencyDetector --window-size 60 --step-size 14 --max-windows 5

# Quick test (3 windows)
python eval/eval_detector.py --detector DarkHoursDetector --max-windows 3

# Comprehensive evaluation (20 windows)
python eval/eval_detector.py --detector FatigueDetector --max-windows 20
```

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--detector` | Detector class name (required) | - |
| `--customer` | Customer name | moprobo |
| `--platform` | Platform name | meta |
| `--window-size` | Window size in days | 30 |
| `--step-size` | Step size between windows | 7 |
| `--max-windows` | Maximum number of windows | 10 |
| `--label-method` | Auto-label method | rule_based |
| `--output-name` | Custom output filename | auto-generated |

**Examples:**

```bash
# Standard 30-day evaluation (10 windows)
python eval/eval_detector.py --detector FatigueDetector

# Long-term 60-day evaluation (5 windows)
python eval/eval_detector.py --detector LatencyDetector --window-size 60 --max-windows 5

# High-resolution evaluation (small windows)
python eval/eval_detector.py --detector DarkHoursDetector --window-size 14 --step-size 3 --max-windows 20
```

**Output:**

Reports are saved to: `src/meta/diagnoser/evaluator/reports/{customer}_sliding/`

### check_detector_satisfaction.py

Check if all detectors meet their target metrics.

**Usage:**

```bash
python eval/check_detector_satisfaction.py
```

**Target Metrics:**

| Detector | F1 Score | Precision | Recall |
|----------|----------|-----------|--------|
| FatigueDetector | ≥0.75 | ≥0.70 | ≥0.80 |
| LatencyDetector | ≥0.75 | ≥0.70 | ≥0.80 |
| DarkHoursDetector | ≥0.75 | ≥0.70 | ≥0.80 |

**Output:**

Shows which detectors meet targets and which need optimization.

## Optimization Scripts

### optimize_detectors.py

Unified detector optimization script with multiple strategies.

**Features:**
- Multiple optimization strategies (agent, manual, adaptive, aggressive)
- Support for single or multiple detectors
- Dry-run mode for testing
- Automatic report backup

**Usage:**

```bash
# Agent-based optimization (default)
python optimize/optimize_detectors.py --strategy agent --detectors FatigueDetector

# Manual optimization with threshold tuning
python optimize/optimize_detectors.py --strategy manual --detectors DarkHoursDetector

# Aggressive optimization (5 cycles)
python optimize/optimize_detectors.py --strategy aggressive --detectors FatigueDetector --cycles 5

# Optimize all detectors
python optimize/optimize_detectors.py --strategy agent --detectors all

# Dry run (no changes)
python optimize/optimize_detectors.py --strategy agent --detectors FatigueDetector --dry-run
```

**Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `agent` | AI agent-based optimization (uses orchestrator) | Full automation |
| `manual` | Manual threshold tuning | Quick fixes |
| `adaptive` | Iterative optimization with auto-adjustment | Balanced approach |
| `aggressive` | Fast 5-cycle optimization | Quick results |

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--strategy` | Optimization strategy | agent |
| `--detectors` | Detectors to optimize | FatigueDetector |
| `--cycles` | Maximum optimization cycles | 10 |
| `--target-f1` | Target F1 score | 0.75 |
| `--target-precision` | Target precision | 0.70 |
| `--target-recall` | Target recall | 0.80 |
| `--use-real-llm` | Use real LLM for agent optimization | False (mock mode) |
| `--dry-run` | Dry run mode | False |

**Examples:**

```bash
# Standard optimization
python optimize/optimize_detectors.py --strategy agent --detectors FatigueDetector

# Aggressive optimization (3 cycles)
python optimize/optimize_detectors.py --strategy aggressive --detectors FatigueDetector --cycles 3

# Optimize all detectors
python optimize/optimize_detectors.py --strategy agent --detectors all

# Dry run to test
python optimize/optimize_detectors.py --strategy agent --detectors FatigueDetector --dry-run
```

## Utility Modules

The `utils/` directory contains shared modules used by all evaluation and optimization scripts:

### data_loader.py

Data loading and preprocessing utilities.

**Functions:**
- `load_moprobo_data()` - Load daily data
- `preprocess_daily_data()` - Preprocess daily data
- `preprocess_hourly_data()` - Preprocess hourly data

**Usage:**

```python
from src.meta.diagnoser.scripts.utils import data_loader

# Load moprobo data
daily_data = data_loader.load_moprobo_data("moprobo", "meta")
```

### sliding_windows.py

Sliding window generation for backtesting.

**Functions:**
- `generate_sliding_windows_daily()` - Generate daily windows
- `generate_sliding_windows_hourly()` - Generate hourly windows

**Usage:**

```python
from src.meta.diagnoser.scripts.utils import sliding_windows

# Generate 10 windows of 30 days
windows = sliding_windows.generate_sliding_windows_daily(
    daily_data,
    window_size_days=30,
    step_days=7,
    max_windows=10
)
```

### evaluation_runner.py

Evaluation execution utilities.

**Functions:**
- `evaluate_detector_on_windows()` - Evaluate on multiple windows
- `evaluate_single_window()` - Evaluate on single window
- `evaluate_multiple_detectors()` - Batch detector evaluation

**Usage:**

```python
from src.meta.diagnoser.scripts.utils import evaluation_runner

# Evaluate detector on windows
results = evaluation_runner.evaluate_detector_on_windows(
    detector=detector,
    windows=windows,
    detector_name="FatigueDetector",
    label_method="rule_based"
)
```

### results_aggregator.py

Results aggregation and summary.

**Functions:**
- `aggregate_results()` - Aggregate across windows
- `calculate_metrics_summary()` - Extract summary metrics
- `format_metrics_for_display()` - Format for display

**Usage:**

```python
from src.meta.diagnoser.scripts.utils import results_aggregator

# Aggregate results
aggregation = results_aggregator.aggregate_results(results, "FatigueDetector")

# Calculate summary
summary = results_aggregator.calculate_metrics_summary(aggregation)

# Format for display
formatted = results_aggregator.format_metrics_for_display(summary)
print(formatted)
```

### metrics_utils.py

Metrics loading and satisfaction checking.

**Functions:**
- `load_metrics()` - Load metrics from reports
- `check_satisfaction()` - Check if targets met
- `format_metrics_report()` - Format metrics for display
- `load_all_metrics()` - Load multiple detector metrics

**Usage:**

```python
from src.meta.diagnoser.scripts.utils import metrics_utils

# Load metrics
metrics = metrics_utils.load_metrics("FatigueDetector")

# Check satisfaction
targets = {"f1_score": 0.75, "precision": 0.70, "recall": 0.80}
all_satisfied, satisfaction = metrics_utils.check_satisfaction(metrics, targets)

# Format report
report = metrics_utils.format_metrics_report("FatigueDetector", metrics, targets)
print(report)
```

## Common Workflows

### Evaluate a Detector

```bash
# 1. Run evaluation
python eval/eval_detector.py --detector FatigueDetector

# 2. Check satisfaction
python eval/check_detector_satisfaction.py
```

### Optimize a Detector

```bash
# 1. Check current performance
python eval/check_detector_satisfaction.py

# 2. Run optimization
python optimize/optimize_detectors.py --strategy agent --detectors FatigueDetector

# 3. Re-evaluate
python eval/eval_detector.py --detector FatigueDetector

# 4. Check satisfaction again
python eval/check_detector_satisfaction.py
```

### Batch Evaluation

```bash
# Evaluate all detectors
python eval/eval_detector.py --detector FatigueDetector
python eval/eval_detector.py --detector LatencyDetector
python eval/eval_detector.py --detector DarkHoursDetector

# Check all at once
python eval/check_detector_satisfaction.py
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the project root:

```bash
# From project root
python src/meta/diagnoser/scripts/eval/eval_detector.py --detector FatigueDetector

# Or use the -m flag
python -m src.meta.diagnoser.scripts.eval.eval_detector --detector FatigueDetector
```

### Data Not Found

If data files are not found, check the path:

```bash
# Default data path
datasets/moprobo/meta/raw/ad_daily_insights_2024-12-17_2025-12-17.csv

# You can specify a different customer/platform
python eval/eval_detector.py --detector FatigueDetector --customer other_customer
```

### Reports Not Saved

If reports are not saved, check the directory exists:

```bash
# Create reports directory if needed
mkdir -p src/meta/diagnoser/evaluator/reports/moprobo_sliding
```

## Migration from Old Scripts

Old scripts have been consolidated into new scripts:

| Old Script | New Script |
|-----------|-----------|
| `evaluate_dark_hours.py` | `eval/eval_detector.py --detector DarkHoursDetector` |
| `evaluate_fatigue.py` | `eval/eval_detector.py --detector FatigueDetector` |
| `evaluate_fatigue_60day.py` | `eval/eval_detector.py --detector FatigueDetector --window-size 60` |
| `evaluate_moprobo.py` | `eval/eval_detector.py --detector FatigueDetector` |
| `evaluate_moprobo_iterative.py` | `eval/eval_detector.py --detector FatigueDetector --max-windows 20` |
| `evaluate_diagnosers.py` | `eval/check_detector_satisfaction.py` |
| `optimize_diagnosers.py` | `optimize/optimize_detectors.py --strategy manual` |
| `run_optimization_5cycles.py` | `optimize/optimize_detectors.py --strategy aggressive --cycles 5` |
| `run_diagnosers.py` | `optimize/optimize_detectors.py --strategy agent --detectors all` |

## Best Practices

1. **Always check satisfaction** before optimizing
2. **Use dry-run mode** to test optimization strategies
3. **Start with aggressive strategy** for quick results
4. **Use agent strategy** for comprehensive optimization
5. **Backup reports** before major changes (automatic in optimize script)
6. **Evaluate after optimization** to verify improvements

## Contributing

When adding new detectors or scripts:

1. Use shared utility modules from `utils/`
2. Follow CLI conventions (argparse, help text)
3. Add examples to this README
4. Test with `--dry-run` first
5. Update detector registries

## Related Documentation

- `../../README.md` - Main diagnoser documentation
- `../../detectors/README.md` - Detector documentation
- `../../evaluator/README.md` - Evaluation system documentation
