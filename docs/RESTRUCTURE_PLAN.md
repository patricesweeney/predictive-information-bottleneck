# Clean Architecture Restructure Plan

## Proposed Directory Structure

```
predictive-information-bottleneck/
├── README.md
├── requirements.txt
├── setup.py                          # NEW: Package installation
├── pyproject.toml                     # NEW: Modern Python project config
│
├── src/
│   └── pib/                          # Main package (predictive-information-bottleneck)
│       ├── __init__.py               # Main public API
│       │
│       ├── batch/                    # 🎯 KEY OUTPUT 1: Batch Analysis
│       │   ├── __init__.py
│       │   ├── analyzer.py           # Main batch analyzer (public API)
│       │   └── implementation.py     # Internal implementation
│       │
│       ├── online/                   # 🎯 KEY OUTPUT 2: Online Analysis  
│       │   ├── __init__.py
│       │   ├── processor.py          # Main online processor (public API)
│       │   └── implementation.py     # Internal implementation
│       │
│       ├── core/                     # Core domain logic
│       │   ├── __init__.py
│       │   ├── interfaces.py         # Domain interfaces/protocols
│       │   ├── models.py             # Domain models/data structures
│       │   └── algorithms/           # Core algorithms
│       │       ├── __init__.py
│       │       ├── information_theory.py
│       │       ├── expectation_maximization.py
│       │       └── state_splitting.py
│       │
│       ├── processes/                # Stochastic process generators
│       │   ├── __init__.py
│       │   ├── generators.py         # Process generators
│       │   └── validation.py         # Process validation
│       │
│       ├── analysis/                 # Analysis utilities
│       │   ├── __init__.py
│       │   ├── empirical.py          # Empirical probability analysis
│       │   └── visualization.py     # Plotting utilities
│       │
│       └── infrastructure/           # Infrastructure concerns
│           ├── __init__.py
│           ├── dependency_injection.py  # DI container
│           └── config.py             # Configuration management
│
├── examples/                         # Usage examples
│   ├── __init__.py
│   ├── batch_analysis_demo.py        # Batch analysis examples
│   ├── online_analysis_demo.py       # Online analysis examples
│   └── process_comparison.py         # Compare multiple processes
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── conftest.py                  # Test configuration
│   ├── unit/                        # Unit tests
│   │   ├── test_batch.py
│   │   ├── test_online.py
│   │   ├── test_core/
│   │   └── test_processes/
│   ├── integration/                 # Integration tests
│   │   ├── test_composition.py
│   │   └── test_end_to_end.py
│   └── fixtures/                    # Test data
│
├── docs/                           # Documentation
│   ├── api/                        # API documentation
│   ├── examples/                   # Example notebooks
│   └── architecture.md            # Architecture documentation
│
└── scripts/                       # Utility scripts
    ├── validate_composition.py    # Architecture validation
    └── benchmark.py              # Performance benchmarks
```

## Key Principles

1. **Clear Entry Points**: `/batch/analyzer.py` and `/online/processor.py` are the main APIs
2. **Layered Architecture**: Core → Analysis → Batch/Online → Examples
3. **Separation of Concerns**: Infrastructure, domain logic, and applications clearly separated
4. **Dependency Direction**: Outer layers depend on inner layers, never reverse
5. **Public API**: Clean `__init__.py` files expose only what's needed

## Migration Benefits

- ✅ **Clear key outputs**: Batch and online analysis prominently featured
- ✅ **Clean imports**: `from pib.batch import analyzer` 
- ✅ **Testable structure**: Easy to test each layer independently
- ✅ **Extensible design**: Easy to add new analysis types
- ✅ **Professional layout**: Follows Python packaging best practices