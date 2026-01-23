# Meta Ads Autopilot - Offline Adset Allocator

Rules-based budget allocation system for Meta Ads adsets using Bayesian-optimized parameters. Analyzes 21 features across ad, adset, campaign, and account levels. The workflow is Extract Features → Tune Parameters → Allocate Budget → Review Results.

## Prerequisites
```bash
# Install Python 3.12
brew install python@3.12

# Install Jupyter kernel
bash scripts/install_jupyter.sh
source ~/.zshrc

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Extract features
python3 run.py extract --customer moprobo

# Budget allocation (rules-based with Bayesian-tuned parameters)
python3 run.py execute --customer moprobo

# Tune rule parameters (Bayesian optimization)
python3 run.py tune --customer moprobo --iterations 100

# Discover patterns from data (decision trees)
python3 run.py discover --customer moprobo

# Get help
python3 run.py --help
python3 run.py {command} --help
```

## Project Structure

```
.
├── run.py                       # Main entry point
├── src/
│   ├── cli/                     # Command implementations
│   │   └── commands/
│   │       ├── extract.py       # Extract CLI
│   │       ├── execute.py       # Allocation CLI
│   │       └── tune.py          # Tuning CLI
│   ├── config/                  # Configuration management
│   │   ├── path_manager.py      # Platform-aware path resolution
│   │   ├── manager.py           # Configuration loading & validation
│   │   └── schemas.py           # Configuration schema definitions
│   ├── adset/                   # Adset budget allocation logic
│   │   ├── allocator/           # Core allocation engine
│   │   │   ├── engine.py        # Main allocator interface
│   │   │   └── rules.py         # Decision rules (safety, performance)
│   │   └── lib/                 # Allocation utilities
│   │       ├── models.py        # Data models (metrics, params)
│   │       ├── decision_rules.py    # Decision logic implementation
│   │       ├── safety_rules.py      # Safety checks (freeze, caps)
│   │       └── decision_rules_helpers.py  # Rule helper functions
│   ├── features/                # Feature extraction pipeline
│   │   ├── core/                # Core extraction logic
│   │   ├── lib/                 # Feature utilities (aggregator, joiner, loader, preprocessor)
│   │   └── utils/               # File discovery, constants, metadata, CSV combiner
│   ├── integrations/            # Third-party integrations
│   │   └── shopify/             # Shopify order data integration
│   │       ├── loader.py        # Shopify CSV loader
│   │       └── features.py      # Shopify feature extraction
│   ├── optimizer/               # Tuning and evaluation
│   │   ├── lib/                 # Bayesian tuners with cross-validation, backtesting
│   │   └── tuning.py            # Allocation simulation & evaluation
│   ├── utils/                   # Utility functions
│   │   ├── customer_paths.py    # Path resolution for customer data (uses PathManager)
│   │   ├── logging_config.py    # Logging configuration
│   │   └── script_helpers.py    # Script utility functions (CLI argument helpers)
│   └── workflows/               # Workflow orchestration
│       ├── base.py              # Base workflow class
│       ├── extract_workflow.py  # Feature extraction workflow
│       ├── tuning_workflow.py   # Tuning workflow
│       └── allocation_workflow.py # Allocation workflow
├── config/                      # Configuration files
│   ├── default.yaml             # Default configuration values
│   └── {customer}/{platform}/   # Customer and platform-specific configs
│       └── rules.yaml           # Allocator configuration (thresholds, weights)
├── datasets/                    # Input data (ad, adset, campaign insights)
│   └── {customer}/{platform}/   # Customer and platform-specific data
│       ├── raw/                 # Raw API responses
│       └── features/            # Extracted features (one file per date)
├── results/                     # Allocation results organized by customer and platform
│   └── {customer}/{platform}/   # Customer and platform-specific results
│       └── rules/               # Rules-based allocation results
└── tests/                       # Unit & integration tests
    ├── unit/                    # Unit tests
    │   ├── adset/               # Allocator tests
    │   │   ├── allocator/       # Core allocator tests
    │   │   └── lib/             # Decision rules tests
    │   ├── features/            # Feature extraction tests
    │   ├── optimizer/           # Tuning tests
    │   └── integrations/        # Integration tests
    │       └── shopify/         # Shopify integration tests
    └── integration/             # Integration tests
        ├── features/            # Feature extraction end-to-end tests
        └── adset/               # Allocator end-to-end tests
```

## Architecture

### Adset Allocator (`src/adset/allocator/`)

**Core Components:**
- **Engine** (`engine.py`): Main allocator interface that orchestrates budget allocation
- **Rules** (`rules.py`): Decision rules for budget adjustments

**Supporting Library** (`src/adset/lib/`):
- **models.py**: Data models for budget allocation (metrics, parameters)
- **decision_rules.py`: Comprehensive rule-based decision logic (42+ rules)
- **safety_rules.py`: Safety checks (freeze on low ROAS, budget caps)
- **decision_rules_helpers.py`: Helper functions for gradient adjustment, trend scaling, health scoring

### Configuration System

**Two-tier configuration:**
1. **`config/default/rules.yaml`**: Default values for all parameters
2. **`config/{customer}/{platform}/rules.yaml`**: Customer/platform-specific overrides

**Configuration sections:**
- `rolling_windows`: Time window settings
- `safety_rules`: Freeze thresholds, budget caps
- `decision_rules`: ROAS thresholds, trend adjustments, multipliers
- `gradient_config`: Gradient-based adjustment parameters
- `trend_config`: Trend scaling configuration
- `health_score_config`: Health score multiplier settings
- `budget_relative_config`: Budget-size-based adjustments
- `sample_size_config`: Confidence-based adjustments by sample size
- `time_weighted_smoothing`: Recent performance weighting
- `monthly_budget`: Monthly budget tracking

### Decision Rules Framework

**Priority-based rule system:**
1. **Safety checks** - Freeze underperforming adsets
2. **Excellent performers** - Aggressive increases for top performers
3. **High performers** - Moderate increases for strong performers
4. **Efficiency rules** - Adjust based on revenue efficiency
5. **Volume rules** - Consider spend, impressions, clicks
6. **Lifecycle rules** - Cold start, learning phase, established
7. **Time-based** - Weekend boosts, Monday recovery, Q4 seasonal
8. **Declining performers** - Decreases for poor performance

**Each rule returns:**
- Adjustment factor (e.g., 1.20 = +20% increase)
- Decision path (e.g., "excellent_roas_rising_trend")

### Shopify Integration

**Purpose:** Validate Meta attribution with actual revenue data

**Components:**
- **`src/integrations/shopify/loader.py`**: Load Shopify order CSVs
- **`src/integrations/shopify/features.py`**: Calculate Shopify ROAS

**Features added:**
- `shopify_roas`: Actual revenue-based ROAS
- `shopify_revenue`: Total Shopify revenue

**Usage:**
```bash
# Place Shopify CSV in: datasets/{customer}/{platform}/raw/shopify.csv
# Features automatically extracted during pipeline
```

### Bayesian Optimization

**Purpose:** Tune 60+ rule parameters to optimize multi-objective goals

**Optimizer:** `src/optimizer/lib/bayesian_tuner.py`
- Uses Gaussian Process Regression
- Time-series cross-validation (prevents overfitting)
- Multi-objective optimization (ROAS, CTR, stability, budget utilization)

**Objectives (`config/{customer}/{platform}/objectives.yaml`):**
- Maximize ROAS
- Maximize CTR
- Minimize budget variance (stability)
- Maximize budget utilization

## Testing

```bash
# Run all tests
make test

# Run unit tests
pytest tests/unit/

# Run specific test module
pytest tests/unit/adset/lib/test_decision_rules.py -v

# Run with coverage
pytest --cov=src/adset --cov-report=html
```

## CI/CD

**GitHub Actions workflows:**
- `.github/workflows/tests.yml`: Run tests on every push
- `.github/workflows/lint.yml`: Run pylint and type checks

**Linting:**
```bash
make lint  # Run pylint
make format  # Format code with black
```

## Development

**Adding new rules:**
1. Add rule method to `src/adset/lib/decision_rules.py`
2. Add configuration to `config/default/rules.yaml`
3. Add test to `tests/unit/adset/lib/test_decision_rules.py`
4. Run Bayesian tuning to optimize parameters

**Adding new integrations:**
1. Create loader in `src/integrations/{platform}/loader.py`
2. Add feature extraction in `src/integrations/{platform}/features.py`
3. Integrate into extraction workflow in `src/cli/commands/extract.py`
4. Add tests in `tests/unit/integrations/{platform}/`

## Contributing

Follow existing code style:
- Use `make format` before committing
- Add tests for new features
- Update README for new functionality
- Run `make lint` and fix warnings

## License

[Add your license here]
