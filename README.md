# Meta Ads Autopilot - Offline Adset Management System

Comprehensive adset management system for Meta Ads combining budget allocation, audience configuration, and creative optimization. Uses rules-based approaches with Bayesian-optimized parameters to analyze 21+ features across ad, adset, campaign, and account levels. The workflow is Extract Features → Tune Parameters → Allocate Budget → Generate Recommendations → Review Results.

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

# Auto-calculate rule parameters from customer data
python3 run.py auto-params --customer moprobo --platform meta

# Run rules-based audience configuration pipeline
python3 run.py rules --customer moprobo --platform meta

# Extract features from images using GPT-4 Vision API
python3 run.py extract-features \
    --top-csv data/top_150_images.csv \
    --bottom-csv data/bottom_150_images.csv

# Extract with real ad performance data
python3 run.py extract-features \
    --ad-data-csv data/ad_data.csv \
    --output-csv data/features_with_roas.csv

# Generate creative recommendations from feature data
python3 run.py recommend \
    --input-csv data/features_with_roas.csv \
    --output-dir config/recommendations/moprobo/meta

# Generate structured prompt from feature recommendations
python3 run.py --customer moprobo --platform taboola prompt structured \
    --base-prompt "A professional product image"

# Generate Nano Banana-optimized prompt via GPT-4o
python3 run.py --customer moprobo --platform taboola prompt nano \
    --base-prompt "A professional product image" \
    --source-image source.jpg

# Generate images with nano-banana-pro model
python3 run.py --customer moprobo --platform taboola generate \
    --source-image product.jpg \
    --prompt "Professional product image" \
    --model nano-banana-pro \
    --num-variations 3

# Run end-to-end pipeline (recommendations → prompts → images)
python3 run.py --customer moprobo --platform taboola run \
    --source-image product.jpg \
    --base-prompt "A professional product image" \
    --num-variations 3

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
│   │       ├── extract.py       # Extract features CLI
│   │       ├── execute.py       # Budget allocation CLI
│   │       ├── tune.py          # Parameter tuning CLI
│   │       ├── auto_params.py    # Auto-calculate parameters CLI
│   │       └── rules.py         # Audience configuration CLI
│   ├── config/                  # Configuration management
│   │   ├── path_manager.py      # Platform-aware path resolution
│   │   ├── manager.py           # Configuration loading & validation
│   │   └── schemas.py           # Configuration schema definitions
│   ├── ad/                      # Ad-level modules
│   │   ├── recommender/          # Creative recommendation engine
│   │   │   ├── features/        # Feature extraction (GPT-4 Vision)
│   │   │   │   ├── extract.py   # Feature extraction and ROAS integration
│   │   │   │   ├── extractors/   # GPT-4 Vision extractor
│   │   │   │   ├── transformers/# Feature transformers
│   │   │   │   └── lib/          # Utilities (parsers, loaders, mergers)
│   │   │   ├── recommendations/ # Recommendation generation
│   │   │   │   ├── rule_engine.py      # Statistical pattern-based recommendations
│   │   │   │   └── prompt_formatter.py # Format recommendations as prompts
│   │   │   └── utils/           # Utilities (api_keys, config_manager, paths, statistics)
│   │   └── generator/            # Creative image generation engine
│   │       ├── core/            # Core functionality
│   │       │   ├── paths.py     # Path utilities (customer/platform support)
│   │       │   ├── prompts/     # Feature-to-prompt conversion
│   │       │   └── generation/  # Image generation via FAL.ai
│   │       ├── orchestrator/    # Prompt building and orchestration
│   │       └── pipeline/        # End-to-end generation pipeline
│   ├── adset/                   # Adset management modules
│   │   ├── allocator/           # Budget allocation engine
│   │   │   └── allocator.py     # Main allocator interface
│   │   ├── generator/           # Audience configuration engine
│   │   │   ├── core/            # Core recommender classes
│   │   │   ├── analyzers/       # Performance analysis
│   │   │   │   ├── advantage_constraints.py
│   │   │   │   ├── opportunity_sizer.py
│   │   │   │   └── shopify_analyzer.py
│   │   │   ├── detection/       # Mistake detection
│   │   │   │   └── mistake_detector.py
│   │   │   ├── generation/      # Recommendation generation
│   │   │   │   ├── audience_aggregator.py
│   │   │   │   ├── audience_recommender.py
│   │   │   │   └── creative_compatibility.py
│   │   │   └── segmentation/    # Audience segmentation
│   │   │       └── segmenter.py
│   │   ├── lib/                 # Shared utilities
│   │   │   ├── models.py        # Data models (metrics, params)
│   │   │   ├── decision_rules.py    # Decision logic implementation
│   │   │   ├── safety_rules.py      # Safety checks (freeze, caps)
│   │   │   └── decision_rules_helpers.py  # Rule helper functions
│   │   ├── features/            # Feature extraction pipeline
│   │   │   ├── core/            # Core extraction logic
│   │   │   ├── lib/             # Feature utilities (aggregator, joiner, loader, preprocessor)
│   │   │   ├── utils/           # File discovery, constants, metadata, CSV combiner
│   │   │   ├── feature_store.py # Adset data loading & preprocessing
│   │   │   └── feature_selector.py # Feature selection
│   │   ├── integrations/        # Third-party integrations
│   │   │   └── shopify/         # Shopify order data integration
│   │   │       ├── loader.py    # Shopify CSV loader
│   │   │       └── features.py  # Shopify feature extraction
│   │   ├── optimizer/           # Tuning and evaluation
│   │   │   ├── lib/             # Bayesian tuners with cross-validation, backtesting
│   │   │   └── tuning.py        # Allocation simulation & evaluation
│   │   ├── workflows/           # Workflow orchestration
│   │   │   ├── base.py          # Base workflow class
│   │   │   ├── extract_workflow.py # Feature extraction workflow
│   │   │   ├── tuning_workflow.py # Tuning workflow
│   │   │   └── allocation_workflow.py # Allocation workflow
│   │   └── budget/              # Budget tracking
│   │       ├── monthly_tracker.py # Monthly budget tracking
│   │       └── state_manager.py   # Budget state management
│   ├── utils/                   # Common utility functions
│   │   ├── customer_paths.py    # Path resolution for customer data (uses PathManager)
│   │   ├── logging_config.py    # Logging configuration
│   │   ├── script_helpers.py    # Script utility functions (CLI argument helpers)
│   │   ├── auto_params.py       # Auto-calculate rule parameters
│   │   ├── target_transformer.py # Target variable transformations
│   │   └── time_series_cv.py    # Time-based cross-validation
│   └── workflows/               # Workflow orchestration
│       ├── base.py              # Base workflow class
│       ├── extract_workflow.py  # Feature extraction workflow
│       ├── tuning_workflow.py   # Tuning workflow
│       └── allocation_workflow.py # Allocation workflow
├── config/                      # Configuration files
│   ├── default/                 # Default configuration values
│   │   ├── development.yaml
│   │   ├── rules.yaml
│   │   └── system.yaml
│   ├── gpt4/                    # GPT-4 Vision API configs
│   │   ├── features.yaml       # GPT-4 feature definitions
│   │   └── prompts.yaml        # GPT-4 prompt templates
│   ├── templates/               # Creative generation templates
│   │   └── {customer}/{platform}/ # Customer/platform templates
│   ├── prompts/                 # Generated prompts
│   │   └── {customer}/{platform}/{date}/{type}/
│   ├── generated/               # Generated images
│   │   └── {customer}/{platform}/{date}/{model}/
│   └── {customer}/{platform}/  # Customer and platform-specific configs
│       └── rules.yaml           # Allocator configuration (thresholds, weights)
│       └── params.yaml          # Generator parameters
│       └── adsets.yaml          # Adset configurations
│       └── recommendations.yaml # Recommendation settings
│       └── objectives.yaml      # Optimization objectives
│       └── generation_config.yaml # Creative generation config
├── datasets/                    # Input data (ad, adset, campaign insights)
│   └── {customer}/{platform}/   # Customer and platform-specific data
│       ├── raw/                 # Raw API responses
│       └── features/            # Extracted features (one file per date)
├── results/                     # Results organized by customer and platform
│   └── {customer}/{platform}/   # Customer and platform-specific results
│       ├── rules/               # Rules-based allocation results
│       └── recommendations/      # Audience recommendations
└── tests/                       # Unit & integration tests
    ├── unit/                    # Unit tests
    │   ├── adset/               # Adset module tests
    │   │   ├── allocator/       # Budget allocator tests
    │   │   ├── generator/        # Audience generator tests
    │   │   ├── lib/             # Decision rules tests
    │   │   ├── features/        # Feature extraction tests
    │   │   ├── optimizer/       # Tuning tests
    │   │   ├── workflows/       # Workflow tests
    │   │   ├── budget/          # Budget tracking tests
    │   │   └── integrations/    # Integration tests
    │   │       └── shopify/     # Shopify integration tests
    │   └── ad/                  # Ad-level module tests
    │       ├── recommender/     # Creative recommender tests
    │       └── generator/        # Creative generator tests
    └── integration/             # Integration tests
        ├── features/            # Feature extraction end-to-end tests
        └── workflows/           # Workflow end-to-end tests
```

## Architecture

### Adset Allocator (`src/adset/allocator/`)
Budget allocation engine that distributes budget across adsets based on performance. Uses 3-layer allocation: safety checks → decision rules → post-modifications.

**Core Components:**
- `allocator.py`: Main allocator interface
- `src/adset/lib/decision_rules.py`: Comprehensive rule-based decision logic (42+ rules)
- `src/adset/lib/safety_rules.py`: Safety checks (freeze on low ROAS, budget caps)
- `src/adset/lib/models.py`: Data models for budget allocation

**Priority-based rule system:**
1. Safety checks - Freeze underperforming adsets
2. Excellent performers - Aggressive increases for top performers
3. High performers - Moderate increases for strong performers
4. Efficiency rules - Adjust based on revenue efficiency
5. Volume rules - Consider spend, impressions, clicks
6. Lifecycle rules - Cold start, learning phase, established
7. Time-based - Weekend boosts, Monday recovery, Q4 seasonal
8. Declining performers - Decreases for poor performance

### Adset Generator (`src/adset/generator/`)
Audience configuration engine that generates audience strategies and configurations.

**Core Components:**
- `core/`: Base recommender classes
- `analyzers/`: Performance analysis (advantage constraints, opportunity sizing, Shopify analysis)
- `detection/`: Mistake detection in audience setup
- `generation/`: Recommendation generation (audience aggregator, recommender, creative compatibility)
- `segmentation/`: Audience segmentation by geo, audience, creative

### Creative Recommender (`src/ad/recommender/`)
Statistical pattern-based creative optimization - analyzes image features to generate ROAS improvement recommendations.

**Philosophy:** "NO AI. NO SPECULATION. JUST STATISTICS."
- Uses statistical pattern detection (not ML models)
- Identifies patterns in top 25% vs bottom 25% performers
- Requires lift >= 1.5x, prevalence >= 10%, statistical significance

**Core Components:**
- `features/`: GPT-4 Vision API feature extraction (116+ features)
- `recommendations/rule_engine.py`: Statistical pattern-based recommendation engine
- `recommendations/prompt_formatter.py`: Format recommendations as prompts

### Creative Generator (`src/ad/generator/`)
Image generation system that converts feature recommendations into optimized prompts and generates images via FAL.ai.

**Core Components:**
- `core/`: Path utilities, prompt conversion, FAL.ai image generation
- `orchestrator/`: Prompt building and orchestration
- `pipeline/`: End-to-end generation pipeline

**Workflow:**
1. Load feature recommendations from creative scorer
2. Convert features to optimized prompts for Nano Banana models
3. Generate images using FAL.ai
4. Validate that generated images match requested features

### Configuration System
Two-tier configuration: `config/default/rules.yaml` (defaults) and `config/{customer}/{platform}/rules.yaml` (overrides).

**Configuration sections:**
- `safety_rules`: Freeze thresholds, budget caps
- `decision_rules`: ROAS thresholds, trend adjustments, multipliers
- `gradient_config`: Gradient-based adjustment parameters
- `trend_config`: Trend scaling configuration
- `health_score_config`: Health score multiplier settings
- `monthly_budget`: Monthly budget tracking

### Bayesian Optimization
Tune 60+ rule parameters to optimize multi-objective goals using Gaussian Process Regression with time-series cross-validation.

**Objectives:** Maximize ROAS, maximize CTR, minimize budget variance (stability), maximize budget utilization.

### Shopify Integration
Validate Meta attribution with actual revenue data. Place Shopify CSV in `datasets/{customer}/{platform}/raw/shopify.csv` - features automatically extracted during pipeline.

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

## Development

**Adding new allocation rules:**
1. Add rule method to `src/adset/lib/decision_rules.py`
2. Add configuration to `config/default/rules.yaml`
3. Add test to `tests/unit/adset/lib/test_decision_rules.py`
4. Run Bayesian tuning to optimize parameters

**Adding new audience generators:**
1. Add generator to `src/adset/generator/`
2. Add configuration to `config/{customer}/{platform}/params.yaml`
3. Add test to `tests/unit/adset/generator/`
4. Validate with historical data

**Adding new integrations:**
1. Create loader in `src/adset/integrations/{platform}/loader.py`
2. Add feature extraction in `src/adset/integrations/{platform}/features.py`
3. Integrate into extraction workflow in `src/cli/commands/extract.py`
4. Add tests in `tests/unit/adset/integrations/{platform}/`

**Adding new creative features:**
1. Add feature definition to `config/gpt4/features.yaml`
2. Update prompt template in `config/gpt4/prompts.yaml` if needed
3. Feature will be automatically extracted by GPT-4 Vision API
4. Patterns will be detected in recommendation engine
5. Add tests in `tests/unit/ad/recommender/`

**Adding new creative generation templates:**
1. Add template to `config/templates/{customer}/{platform}/generation_config.yaml`
2. Update prompt converter in `src/ad/generator/core/prompts/` if needed
3. Add feature mapping in `src/ad/generator/orchestrator/feature_mapper.py` if needed
4. Add tests in `tests/unit/ad/generator/`
