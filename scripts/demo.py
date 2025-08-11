"""
Information Bottleneck Analysis Demonstrations
============================================

Consolidated demonstrations showing batch and online analysis capabilities.
"""

import numpy as np
from typing import Dict, List, Optional
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from processes import PROCESS_GENERATORS
from batch import run_default_analysis
from online import run_online_analysis


def demo_batch_analysis(selected_processes: Optional[List[str]] = None):
    """
    Demonstrate batch information bottleneck analysis.
    """
    print("=== BATCH INFORMATION BOTTLENECK ANALYSIS ===")
    
    if selected_processes is None:
        selected_processes = ["Golden-Mean", "IID(p=0.5)", "Even"]
    
    print(f"Analyzing processes: {selected_processes}")
    
    # Run analysis
    analyzer, results = run_default_analysis(selected_processes=selected_processes)
    
    # Display results
    for process_name, result in results.items():
        print(f"\n{process_name}:")
        print(f"  Final states: {result['num_states'][-1]}")
        print(f"  Final free energy: {result['free_energies'][-1]:.3f}")
        print(f"  Empirical MI: {result['empirical_mutual_information']:.3f}")
    
    # Create plots including VFE decomposition
    print("\nCreating plots...")
    analyzer.create_information_bottleneck_plot()
    analyzer.create_phase_transition_plot()
    analyzer.create_vfe_decomposition_plot()
    
    return analyzer, results


def demo_online_analysis(process_name: str = "Golden-Mean", sequence_length: int = 5000):
    """
    Demonstrate online information bottleneck processing.
    """
    print("=== ONLINE INFORMATION BOTTLENECK PROCESSING ===")
    
    if process_name not in PROCESS_GENERATORS:
        print(f"Process '{process_name}' not found. Available: {list(PROCESS_GENERATORS.keys())}")
        return None, None
    
    print(f"Processing {sequence_length} symbols from {process_name}")
    
    # Run online analysis
    processor, results = run_online_analysis(
        PROCESS_GENERATORS[process_name],
        sequence_length=sequence_length,
        process_name=process_name
    )
    
    # Display results
    print(f"\nResults:")
    print(f"  Final states: {results['current_num_states']}")
    print(f"  Total contexts observed: {results['total_contexts']}")
    print(f"  State splits occurred at: {results['split_times']}")
    
    # Create plots
    print("\nCreating plots...")
    processor.create_online_analysis_plot()
    
    return processor, results


def demo_process_comparison():
    """
    Compare different stochastic processes.
    """
    print("=== PROCESS COMPARISON ===")
    
    comparison_processes = ["IID(p=0.5)", "Golden-Mean", "Even", "Thue-Morse"]
    
    # Generate sample sequences
    sequence_length = 1000
    
    print(f"Generating {sequence_length} symbols from each process:")
    for process_name in comparison_processes:
        if process_name in PROCESS_GENERATORS:
            generator = PROCESS_GENERATORS[process_name]
            sequence = generator(sequence_length, seed=42)
            
            # Basic statistics
            mean_value = np.mean(sequence)
            transitions = np.sum(np.diff(sequence) != 0)
            
            print(f"  {process_name:>20}: mean={mean_value:.3f}, transitions={transitions}")
            
            # Show first 50 symbols
            print(f"    First 50: {sequence[:50]}")
    
    return True


def demo_custom_analysis():
    """
    Demonstrate custom analysis using the building blocks.
    """
    print("=== CUSTOM ANALYSIS USING BUILDING BLOCKS ===")
    
    from implementations import StandardAnalysisFactory
    
    # Create factory
    factory = StandardAnalysisFactory()
    
    # Build custom analyzer with specific parameters
    analyzer = factory.create_batch_analyzer(
        factory.create_probability_analyzer(),
        factory.create_information_calculator(),
        factory.create_em_optimizer(factory.create_information_calculator()),
        factory.create_state_splitter(
            factory.create_information_calculator(),
            factory.create_em_optimizer(factory.create_information_calculator())
        )
    )
    
    # Custom configuration
    analyzer.past_window_length = 8
    analyzer.future_window_length = 3
    analyzer.sample_length = 50000
    analyzer.beta_schedule = np.geomspace(0.1, 50, 50)
    analyzer.maximum_states_allowed = 8
    
    print("Running custom analysis with:")
    print(f"  Past window: {analyzer.past_window_length}")
    print(f"  Future window: {analyzer.future_window_length}")
    print(f"  Sample length: {analyzer.sample_length}")
    print(f"  Beta values: {len(analyzer.beta_schedule)}")
    print(f"  Max states: {analyzer.maximum_states_allowed}")
    
    # Analyze Golden Mean process
    result = analyzer.run_analysis(PROCESS_GENERATORS["Golden-Mean"], "Golden-Mean")
    
    print(f"\nCustom analysis results:")
    print(f"  Final states: {result['num_states'][-1]}")
    print(f"  Final complexity: {result['complexities'][-1]:.3f}")
    print(f"  Final accuracy: {result['accuracies'][-1]:.3f}")
    
    return analyzer, result


def interactive_demo():
    """
    Interactive demo with menu.
    """
    print("\n" + "="*60)
    print("PREDICTIVE INFORMATION BOTTLENECK ANALYSIS")
    print("="*60)
    
    while True:
        print("\nAvailable demonstrations:")
        print("1. Batch Analysis")
        print("2. Online Analysis") 
        print("3. Process Comparison")
        print("4. Custom Analysis")
        print("5. Show Plots")
        print("6. Quit")
        
        try:
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == "1":
                demo_batch_analysis()
            elif choice == "2":
                process_name = input(f"Process name ({list(PROCESS_GENERATORS.keys())}): ").strip()
                if not process_name:
                    process_name = "Golden-Mean"
                demo_online_analysis(process_name)
            elif choice == "3":
                demo_process_comparison()
            elif choice == "4":
                demo_custom_analysis()
            elif choice == "5":
                from visualization import show_all_plots
                show_all_plots()
            elif choice == "6":
                break
            else:
                print("Invalid choice. Please select 1-6.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Check if running interactively
    import sys
    
    if len(sys.argv) > 1:
        demo_type = sys.argv[1].lower()
        if demo_type == "batch":
            demo_batch_analysis()
        elif demo_type == "online":
            demo_online_analysis()
        elif demo_type == "comparison":
            demo_process_comparison()
        elif demo_type == "custom":
            demo_custom_analysis()
        else:
            print(f"Unknown demo type: {demo_type}")
            print("Available: batch, online, comparison, custom")
    else:
        # Interactive mode
        interactive_demo()
    
    # Show all plots at the end (save to figures directory)
    from visualization import show_all_plots
    show_all_plots(save_to_dir="results/figures")