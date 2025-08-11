# Predictive Information Bottleneck: Modular Implementation

A modular implementation of epsilon machines and information bottleneck methods for analyzing stochastic processes. Extracted and refactored from the original Jupyter notebook into composable Python modules following "Structure and Interpretation of Computer Programs" design principles.

## Overview

This package provides tools for:
- **Stochastic Process Generation**: IID, Markov, sofic, and complex deterministic processes
- **Empirical Analysis**: Extract conditional probabilities from sequence data  
- **Information Bottleneck**: Optimize compression-prediction trade-offs
- **Automatic State Splitting**: Eigenvalue-based adaptive model growth
- **Batch Analysis**: Comprehensive analysis across multiple processes
- **Online Learning**: Real-time streaming information bottleneck
- **Validation**: Sanity checking and process verification

## Design Philosophy

- **Plain English naming**: `extract_empirical_future_given_past()` instead of cryptic abbreviations
- **Modular composition**: Each module works independently or together
- **Functional design**: Clear interfaces with minimal side effects
- **Comprehensive documentation**: Every function thoroughly documented
- **SICP-style**: Structure and Interpretation of Computer Programs principles

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or manually install core dependencies
pip install numpy matplotlib

# Optional: For k-means initialization
pip install scikit-learn
```

## Quick Start

### Basic Usage

```python
from src import PROCESS_GENERATORS, run_default_analysis

# Generate sequences from different processes
generator = PROCESS_GENERATORS["Golden-Mean"]
sequence = generator(10000, seed=42)

# Run comprehensive batch analysis
analysis = run_default_analysis(create_plots=True)
```

### Interactive Demo

```python
# Run the main demonstration script
python src/main_demo.py --interactive
```

## Module Structure

```
src/
├── __init__.py                        # Package initialization
├── stochastic_processes.py            # Process generators
├── empirical_analysis.py              # Probability extraction
├── information_theory.py              # Core IT computations
├── expectation_maximization.py        # EM algorithm
├── state_splitting.py                 # Automatic state growth
├── batch_information_bottleneck.py    # Batch analysis framework
├── online_information_bottleneck.py   # Streaming analysis
├── process_validation.py              # Validation and testing
└── main_demo.py                       # Comprehensive demonstrations
```

## Stochastic Processes

The package includes generators for various process types:

- **IID Processes**: `IID(p=0.5)`, `IID(p=0.25)`
- **Markov Chains**: `Two-state Markov`
- **Sofic Processes**: `Golden-Mean`, `Even`, `No-Three-Ones`
- **Complex Processes**: `Alternating Geom p=0.3`, `Thue-Morse`

Each process has different statistical complexity (Cμ) ranging from 0 bits to infinite.

## Examples

### Analyze a Single Process

```python
from src.batch_information_bottleneck import BatchInformationBottleneckAnalysis
from src.stochastic_processes import PROCESS_GENERATORS

# Create analysis pipeline
analysis = BatchInformationBottleneckAnalysis(
    past_window_length=8,
    future_window_length=2,
    sample_length=50000
)

# Analyze Golden Mean process
generator = PROCESS_GENERATORS["Golden-Mean"]
results = analysis.run_analysis_for_process("Golden-Mean", generator)

# Create plots
analysis.create_information_bottleneck_plot()
```

### Online/Streaming Analysis

```python
from src.online_information_bottleneck import OnlineInformationBottleneck
from src.stochastic_processes import PROCESS_GENERATORS

# Create online analyzer
online_ib = OnlineInformationBottleneck(
    past_window_length=20,
    future_window_length=4,
    inverse_temperature_beta=60.0,
    learning_rate=0.05
)

# Generate and process streaming data
generator = PROCESS_GENERATORS["Thue-Morse"]
sequence = generator(30000, seed=1)

online_ib.process_sequence(sequence)
online_ib.create_online_plots()
```

### Process Validation

```python
from src.process_validation import run_comprehensive_validation
from src.stochastic_processes import PROCESS_GENERATORS

# Validate all processes
results = run_comprehensive_validation(PROCESS_GENERATORS)
print_validation_report(results)
```

## Command Line Interface

The main demo script provides various entry points:

```bash
# Interactive menu
python src/main_demo.py --interactive

# Run specific demonstration
python src/main_demo.py --demo 5  # Batch analysis

# Quick analysis
python src/main_demo.py --quick-batch    # All processes
python src/main_demo.py --quick-online   # Online demo

# Validation only
python src/main_demo.py --validate
```

## Information Bottleneck Theory

The information bottleneck method finds optimal trade-offs between:

- **Compression**: Minimize complexity I(Past; State)
- **Prediction**: Maximize accuracy E[log p(Future|State)]

The variational free energy F = Complexity - β × Accuracy balances these objectives through the inverse temperature parameter β.

### Key Concepts

- **β → 0**: Favors compression (fewer states)
- **β → ∞**: Favors accuracy (more states)
- **State Splitting**: Automatic growth when β × λ_max > 1
- **Phase Transitions**: Discontinuous changes in state count

## Advanced Features

### Automatic State Splitting

Uses eigenvalue analysis of the Hessian matrix to detect when states become linearly unstable and should be split.

### Exponential Forgetting

Online learning can use exponential forgetting to adapt to non-stationary processes.

### Comprehensive Validation

Built-in validation checks process-specific properties:
- IID: Symbol probabilities
- Sofic: Forbidden word constraints  
- Markov: Memory dependence
- Complex: Pattern diversity

## Contributing

The modular design makes it easy to extend:

1. **New Processes**: Add generators to `stochastic_processes.py`
2. **New Analysis**: Extend the batch or online frameworks
3. **New Validation**: Add checks to `process_validation.py`
4. **New Visualizations**: Extend the plotting methods

## References

- Crutchfield, J.P. & Young, K. (1989). Inferring statistical complexity. Physical Review Letters.
- Tishby, N., Pereira, F.C., Bialek, W. (1999). The information bottleneck method.
- Still, S., Crutchfield, J.P., Bell, C.J. (2012). Optimal causal inference.

## License

This implementation is derived from the epsilon machines notebook and follows the same principles of open scientific computing.