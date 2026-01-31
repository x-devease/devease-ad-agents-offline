# Ads Autopilot

Comprehensive ad management system for Meta Ads with 5 major components:

1. **Ad Miner** (`src/meta/ad/miner/`) - Creative recommendation engine using statistical pattern detection
2. **Ad Generator** (`src/meta/ad/generator/`) - Creative image generation using FAL.ai
3. **Adset Allocator** (`src/meta/adset/allocator/`) - Budget allocation engine with Bayesian-optimized parameters
4. **Adset Generator** (`src/meta/adset/generator/`) - Audience configuration engine with historical validation
5. **Nano Banana Pro Agent** (`src/agents/nano/`) - Prompt enhancement agent for high-fidelity creative generation

The system uses rules-based approaches to analyze 21+ features across ad, adset, campaign, and account levels. Workflow: Enhance Prompts → Extract Creative Features → Generate Recommendations → Generate Images → Generate Audience Configs → Allocate Budget.

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
# ===== Nano Banana Pro Agent - Prompt Enhancement =====

# Transform generic prompt into high-fidelity NB Pro prompt (simple mode)
python3 run.py nano enhance --prompt "Create an ad for our mop"

# Full enhancement with metadata and thinking block
python3 run.py nano full --prompt "Show our mop in action" --enable-thinking

# Run predefined examples (1-5)
python3 run.py nano example --example 1

# Run unit tests
python3 run.py nano test

# ===== Adset Allocator - Budget Management =====

# Extract features from raw Meta ads data
python3 run.py extract --customer {customer}

# Budget allocation (rules-based with Bayesian-tuned parameters)
python3 run.py execute --customer {customer}

# Tune rule parameters (Bayesian optimization)
python3 run.py tune --customer {customer} --iterations 100

# Discover patterns from data (decision trees)
python3 run.py discover --customer {customer}

# ===== Adset Generator - Audience Configuration =====

# Run rules-based audience configuration pipeline
python3 run.py rules --customer {customer} --platform {platform}

# ===== Ad Miner - Creative Intelligence =====

# Extract features from images using GPT-4 Vision API
python3 run.py extract-features --ad-data-csv data/ad_data.csv --output-csv data/features_with_roas.csv

# Generate creative recommendations from feature data
python3 run.py recommend --input-csv data/features_with_roas.csv --output-dir config/recommendations/{customer}/{platform}

# ===== Ad Generator - Creative Production =====

# Generate prompts from recommendations
python3 run.py --customer {customer} --platform {platform} prompt structured --base-prompt "A professional product image"

# Generate images with nano-banana models
python3 run.py --customer {customer} --platform {platform} generate --source-image product.jpg --prompt "Professional product image" --num-variations 3

# Run end-to-end pipeline (recommendations → prompts → images)
python3 run.py --customer {customer} --platform {platform} run --source-image product.jpg --base-prompt "A professional product image" --num-variations 3

# ===== General =====

# Get help
python3 run.py --help
python3 run.py {command} --help
```

## Project Structure

```
.
├── run.py                       # Main entry point
├── src/
│   ├── agents/                  # AI Agents
│   │   └── nano/                # Nano Banana Pro Prompt Enhancement Agent
│   │       ├── core/            # Core agent functionality
│   │       │   ├── agent.py     # Main PromptEnhancementAgent class
│   │       │   ├── types.py     # Data classes and types
│   │       │   ├── context_enrichment.py # Product/brand context enrichment
│   │       │   ├── thinking_engine.py # Strategic thinking & technique selection
│   │       │   └── quality_verifier.py # Quality checks
│   │       ├── parsers/         # Input parsing
│   │       │   └── input_parser.py # Intent detection & categorization
│   │       ├── techniques/      # NB Pro techniques
│   │       │   └── orchestrator.py # Applies 8 NB Pro techniques
│   │       ├── formatters/      # Output formatting
│   │       │   ├── natural_language_builder.py # Conversational prompt building
│   │       │   ├── technical_specs.py # Resolution, lighting, camera specs
│   │       │   ├── guards.py    # Anti-hallucination constraints
│   │       │   └── output_formatter.py # Final assembly
│   │       ├── examples.py      # 5 usage examples
│   │       └── __init__.py      # Public API exports
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
├── config/                      # Configuration files (shared across miner, generator, reviewer)
│   ├── {customer}/{platform}/   # Shared customer/platform config
│   │   ├── config.yaml          # SHARED config (miner + generator + reviewer)
│   │   └── patterns.yaml        # Ad miner output patterns
│   ├── agents/                  # Agent configurations
│   │   └── nano/                # Nano Banana Pro Agent config
│   │       └── nano_config.yaml # Agent configuration, product & brand databases
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
│           ├── psychology_catalog.yaml # System: 14 psychology types
│           ├── text_templates.yaml     # System: Text overlay templates
│           ├── templates/      # Generation templates
│           │   └── {customer}/{platform}/
│           ├── prompts/        # Generated prompts (output)
│           │   └── {customer}/{platform}/{date}/{type}/
│           └── generated/      # Generated images (output)
│               └── {customer}/{platform}/{date}/{model}/
├── datasets/                    # Input data (ad, adset, campaign insights)
│   └── {customer}/{platform}/   # Customer and platform-specific data
│       ├── raw/                 # Raw API responses
│       └── features/            # Extracted features (one file per date)
├── results/                     # Results organized by customer and platform
│   └── {customer}/{platform}/   # Customer and platform-specific results
│       ├── rules/               # Rules-based allocation results
│       └── recommendations/      # Audience recommendations
└── tests/                       # Unit & integration tests (all at root level)
    ├── unit/                    # Unit tests (44+ test files)
    │   ├── agents/              # Agent tests
    │   │   └── nano/            # Nano agent tests
    │   ├── ad/                  # Ad-level module tests
    │   │   └── miner/           # Ad miner tests
    │   ├── adset/               # Adset module tests
    │   │   ├── allocator/       # Budget allocator tests
    │   │   ├── generator/       # Audience generator tests
    │   │   ├── lib/             # Decision rules tests
    │   │   ├── features/        # Feature extraction tests
    │   │   ├── optimizer/       # Tuning tests
    │   │   ├── workflows/       # Workflow tests
    │   │   ├── budget/          # Budget tracking tests
    │   │   └── integrations/    # Integration tests
    │   │       └── shopify/     # Shopify integration tests
    │   ├── budget/              # Budget tracking tests
    │   ├── config/              # Config management tests
    │   ├── features_store/      # Feature store tests
    │   ├── integrations/        # Third-party integrations
    │   ├── recommendations/     # Recommendation engine tests
    │   ├── tools/               # Tool tests
    │   ├── utils_auto/          # Auto-parameter utility tests
    │   └── workflows/           # Workflow tests
    └── integration/             # Integration tests (12 test files)
        ├── ad/                  # Ad module integration tests
        │   └── miner/           # Ad miner integration tests
        ├── adset/               # Adset integration tests
        ├── features/            # Feature extraction end-to-end tests
        └── workflows/           # Workflow end-to-end tests
```
