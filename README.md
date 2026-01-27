# Ads Autopilot

Comprehensive ad management system for Meta Ads with 4 major components:

1. **Ad Miner** (`src/meta/ad/miner/`) - Creative recommendation engine using statistical pattern detection
2. **Ad Generator** (`src/meta/ad/generator/`) - Creative image generation using FAL.ai
3. **Adset Allocator** (`src/meta/adset/allocator/`) - Budget allocation engine with Bayesian-optimized parameters
4. **Adset Generator** (`src/meta/adset/generator/`) - Audience configuration engine with historical validation

The system uses rules-based approaches to analyze 21+ features across ad, adset, campaign, and account levels. Workflow: Extract Creative Features → Generate Recommendations → Generate Images → Generate Audience Configs → Allocate Budget.

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
python3 run.py extract --customer {customer}

# Budget allocation (rules-based with Bayesian-tuned parameters)
python3 run.py execute --customer {customer}

# Tune rule parameters (Bayesian optimization)
python3 run.py tune --customer {customer} --iterations 100

# Discover patterns from data (decision trees)
python3 run.py discover --customer {customer}

# Run rules-based audience configuration pipeline
python3 run.py rules --customer {customer} --platform {platform}

# Extract features from images using GPT-4 Vision API
python3 run.py extract-features --ad-data-csv data/ad_data.csv --output-csv data/features_with_roas.csv

# Generate creative recommendations from feature data
python3 run.py recommend --input-csv data/features_with_roas.csv --output-dir config/recommendations/{customer}/{platform}

# Generate prompts from recommendations
python3 run.py --customer {customer} --platform {platform} prompt structured --base-prompt "A professional product image"

# Generate images with nano-banana models
python3 run.py --customer {customer} --platform {platform} generate --source-image product.jpg --prompt "Professional product image" --num-variations 3

# Run end-to-end pipeline (recommendations → prompts → images)
python3 run.py --customer {customer} --platform {platform} run --source-image product.jpg --base-prompt "A professional product image" --num-variations 3

# Get help
python3 run.py --help
python3 run.py {command} --help
```

## Project Structure

```
.
├── run.py                       # Main entry point
├── src/
│   ├── config/                  # Configuration management (shared)
│   │   ├── path_manager.py      # Platform-aware path resolution
│   │   ├── manager.py           # Configuration loading & validation
│   │   └── schemas.py           # Configuration schema definitions
│   ├── meta/                    # Meta Ads modules
│   │   ├── ad/                  # Ad-level modules
│   │   │   ├── recommender/     # Creative recommendation engine
│   │   │   │   ├── features/    # Feature extraction (GPT-4 Vision)
│   │   │   │   │   ├── extract.py # Feature extraction and ROAS integration
│   │   │   │   │   ├── extractors/ # GPT-4 Vision extractor
│   │   │   │   │   ├── transformers/ # Feature transformers
│   │   │   │   │   └── lib/     # Utilities (parsers, loaders, mergers)
│   │   │   │   ├── recommendations/ # Recommendation generation
│   │   │   │   │   ├── rule_engine.py # Statistical pattern-based recommendations
│   │   │   │   │   └── prompt_formatter.py # Format recommendations as prompts
│   │   │   │   └── utils/       # Utilities (api_keys, config_manager, paths, statistics)
│   │   │   └── generator/       # Creative image generation engine
│   │   │       ├── core/        # Core functionality
│   │   │       │   ├── paths.py # Path utilities (customer/platform support)
│   │   │       │   ├── prompts/ # Feature-to-prompt conversion
│   │   │       │   └── generation/ # Image generation via FAL.ai
│   │   │       ├── orchestrator/ # Prompt building and orchestration
│   │   │       └── pipeline/    # End-to-end generation pipeline
│   │   └── adset/               # Adset management modules
│   │   ├── allocator/           # Budget allocation engine
│   │   │   ├── allocator.py     # Main allocator interface
│   │   │   ├── budget/          # Budget tracking
│   │   │   │   ├── monthly_tracker.py # Monthly budget tracking
│   │   │   │   └── state_manager.py   # Budget state management
│   │   │   ├── lib/             # Allocator utilities
│   │   │   │   ├── models.py        # Data models (metrics, params)
│   │   │   │   ├── decision_rules.py    # Decision logic implementation
│   │   │   │   ├── safety_rules.py      # Safety checks (freeze, caps)
│   │   │   │   └── decision_rules_helpers.py  # Rule helper functions
│   │   │   ├── optimizer/       # Parameter tuning
│   │   │   │   ├── lib/         # Bayesian tuners with cross-validation, backtesting
│   │   │   │   └── tuning.py    # Allocation simulation & evaluation
│   │   │   ├── utils/           # Allocator utilities
│   │   │   │   ├── parser.py    # Config parser
│   │   │   │   ├── helpers.py   # Helper functions
│   │   │   │   └── summary.py   # Summary functions
│   │   │   └── workflows/       # Allocator workflows
│   │   │       ├── allocation_workflow.py # Allocation workflow
│   │   │       └── tuning_workflow.py # Tuning workflow
│   │   ├── cli/                 # CLI commands (adset-specific)
│   │   │   └── commands/
│   │   │       ├── extract.py       # Extract features CLI
│   │   │       ├── execute.py       # Budget allocation CLI
│   │   │       ├── tune.py          # Parameter tuning CLI
│   │   │       ├── auto_params.py   # Auto-calculate parameters CLI
│   │   │       └── rules.py         # Audience configuration CLI
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
│   │   └── features/            # Feature extraction pipeline (shared)
│   │       ├── core/            # Core extraction logic
│   │       ├── lib/             # Feature utilities (aggregator, joiner, loader, preprocessor)
│   │       ├── utils/           # File discovery, constants, metadata, CSV combiner
│   │       ├── integrations/    # Third-party integrations
│   │       │   └── shopify/     # Shopify order data integration
│   │       │       ├── loader.py    # Shopify CSV loader
│   │       │       └── features.py  # Shopify feature extraction
│   │       ├── workflows/       # Feature extraction workflows
│   │       │   ├── base.py          # Base workflow class
│   │       │   └── extract_workflow.py # Feature extraction workflow
│   │       ├── feature_store.py # Adset data loading & preprocessing
│   │       └── feature_selector.py # Feature selection
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
├── config/                      # Configuration files (mirrors src/ structure)
│   ├── adset/                   # Adset module configs
│   │   ├── allocator/          # Budget allocation configs
│   │   │   ├── rules.yaml      # Default allocator rules
│   │   │   ├── system.yaml     # System defaults
│   │   │   ├── development.yaml # Development environment
│   │   │   └── {customer}/{platform}/ # Customer/platform configs
│   │   │       ├── rules.yaml  # Allocator rules (thresholds, weights)
│   │   │       └── objectives.yaml # Optimization objectives
│   │   └── generator/          # Audience configuration configs
│   │       └── {customer}/{platform}/ # Customer/platform configs
│   │           ├── params.yaml # Generator parameters
│   │           ├── adsets.yaml # Adset configurations
│   │           └── recommendations.yaml # Recommendation settings
│   └── ad/                     # Ad module configs
│       ├── recommender/        # Creative recommender configs
│       │   ├── gpt4/           # GPT-4 Vision API configs
│       │   │   ├── features.yaml # GPT-4 feature definitions
│       │   │   └── prompts.yaml # GPT-4 prompt templates
│       │   └── recommendations/ # Recommendation outputs
│       │       └── {customer}/{platform}/{date}/
│       └── generator/          # Creative generator configs
│           ├── templates/      # Generation templates
│           │   └── {customer}/{platform}/
│           ├── prompts/        # Generated prompts
│           │   └── {customer}/{platform}/{date}/{type}/
│           ├── generated/      # Generated images
│           │   └── {customer}/{platform}/{date}/{model}/
│           └── {customer}/{platform}/ # Customer/platform configs
│               └── generation_config.yaml # Generation config
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
