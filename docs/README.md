# Documentation

This directory contains all project documentation for the Predictive Information Bottleneck analysis library.

## Architecture Documentation

### [MODULAR_STRUCTURE.md](MODULAR_STRUCTURE.md)
- Original modular structure documentation
- Describes the 9-module decomposition from the notebook
- Shows SICP-style composition examples
- Historical reference for the initial architecture

### [RESTRUCTURE_PLAN.md](RESTRUCTURE_PLAN.md)  
- Clean architecture restructure proposal
- Proposed directory structure with nested modules
- Design principles and migration benefits
- Blueprint for the pib/ subdirectory approach

### [CONSOLIDATION_SUMMARY.md](CONSOLIDATION_SUMMARY.md)
- **Current state documentation** 
- Details the final simplified architecture
- Shows reduction from 36 → 11 files
- Explains what was preserved vs eliminated

### [VISUALIZATION_SEPARATION.md](VISUALIZATION_SEPARATION.md)
- Latest architectural improvement
- Documents separation of plotting from analysis logic
- Details the new `visualization.py` module
- Shows clean separation of concerns

## Architecture Evolution

The documentation shows the evolution of the project architecture:

1. **Monolithic Notebook** → **9 Modular Files** (MODULAR_STRUCTURE.md)
2. **9 Files** → **Complex Nested Structure** (RESTRUCTURE_PLAN.md) 
3. **Complex Structure** → **Simple 11 Files** (CONSOLIDATION_SUMMARY.md)
4. **Mixed Concerns** → **Separated Visualization** (VISUALIZATION_SEPARATION.md)

## Current State

The final architecture (documented in CONSOLIDATION_SUMMARY.md and VISUALIZATION_SEPARATION.md) represents the "right abstraction" - minimal hierarchy that supports:

- ✅ Dependency injection
- ✅ Composability  
- ✅ Modularity
- ✅ Clean separation of concerns
- ✅ Maintainable codebase

## API Documentation

See the `api/` directory for detailed API documentation (if generated).

## Examples

See the `examples/` directory for usage examples and tutorials.