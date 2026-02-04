# Diagnoser Scripts

This directory contains all scripts for evaluating, optimizing, and testing the diagnoser agents.

## Evaluation Scripts

### Single-Detector Evaluation
- **`evaluate_fatigue.py`** - Evaluates FatigueDetector on sliding windows
- **`evaluate_dark_hours.py`** - Evaluates DarkHoursDetector on sliding windows
- **`evaluate_latency.py`** - (Coming soon) Evaluates LatencyDetector on sliding windows

### Multi-Detector Evaluation
- **`evaluate_diagnosers.py`** - Checks satisfaction of all detectors against target metrics
- **`evaluate_moprobo.py`** - Evaluates all detectors on moprobo dataset
- **`evaluate_moprobo_iterative.py`** - Iterative evaluation with detailed analysis
- **`test_diagnosers_moprobo.py`** - Quick test on moprobo sample data
- **`evaluate_fatigue_60day.py`** - FatigueDetector evaluation on 60-day data

### Quick Helpers
- **`quick_eval_fatigue.py`** - Quick fatigue evaluation with optimized thresholds

## Optimization Scripts

### Config-Based Optimization (Recommended)
- **`optimize_diagnosers.py`** - Multi-cycle optimization that writes thresholds to config files
  - Updates: `src/meta/diagnoser/detectors/config/*_detector_config.json`
  - Safe: Doesn't modify source code
  - Reversible: Can restore previous configs

### Source-Code Optimization (Aggressive)
- **`run_optimization_5cycles.py`** - 5-cycle aggressive optimization
  - Updates: DEFAULT_THRESHOLDS in source `.py` files
  - Direct: Modifies detector source code
  - Use with caution: Changes are committed to git

## Agent Orchestration

- **`run_diagnosers.py`** - Runs the full diagnoser agent orchestrator
  - Uses PM, Coder, Reviewer, Judge, Memory agents
  - Autonomous optimization through LLM agents
  - Experimental: Uses mock LLM mode by default

## Usage Examples

### Check if detectors are satisfied
```bash
PYTHONPATH=$(pwd) python3 src/meta/diagnoser/scripts/evaluate_diagnosers.py
```

### Run single detector evaluation
```bash
PYTHONPATH=$(pwd) python3 src/meta/diagnoser/scripts/evaluate_fatigue.py
```

### Run multi-cycle optimization (config-based)
```bash
PYTHONPATH=$(pwd) python3 src/meta/diagnoser/scripts/optimize_diagnosers.py
```

### Run aggressive optimization (source-code editing)
```bash
PYTHONPATH=$(pwd) python3 src/meta/diagnoser/scripts/run_optimization_5cycles.py
```

### Test on moprobo data
```bash
PYTHONPATH=$(pwd) python3 src/meta/diagnoser/scripts/test_diagnosers_moprobo.py
```

## Target Metrics

All detectors are evaluated against these targets:
- **Precision**: ≥ 0.70 (avoid false alarms)
- **Recall**: ≥ 0.80 (catch real problems)
- **F1-Score**: ≥ 0.75 (balanced performance)

## Output Locations

- **Evaluation Reports**: `src/meta/diagnoser/judge/reports/moprobo_sliding/`
  - `fatigue_sliding_10windows.json`
  - `latency_sliding_10windows.json`
  - `dark_hours_sliding_10windows.json`

- **Detector Configs**: `src/meta/diagnoser/detectors/config/`
  - `fatigue_detector_config.json`
  - `dark_hours_detector_config.json`
  - `latency_detector_config.json`

## Note on Optimization Approaches

The codebase includes two optimization approaches:

1. **Config-Based (Recommended)**: `optimize_diagnosers.py`
   - Writes to separate config files
   - Easier to track changes
   - Doesn't modify source code
   - Can rollback easily

2. **Source-Based (Aggressive)**: `run_optimization_5cycles.py`
   - Edits DEFAULT_THRESHOLDS directly
   - Changes become part of code
   - Faster iterations
   - Better for production deployment

Choose based on your needs:
- **Development/Testing**: Use config-based optimization
- **Production Deployment**: Use source-based optimization (then commit)
