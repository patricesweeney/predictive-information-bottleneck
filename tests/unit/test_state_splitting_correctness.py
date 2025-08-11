"""
State Splitting Correctness Tests
=================================

Comprehensive validation that state splitting discovers the correct
causal structure for known processes with theoretical ground truth.
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from processes import PROCESS_GENERATORS
from empirical_analysis import extract_empirical_future_given_past
from state_splitting import run_adaptive_state_growth, analyze_state_stability
from interfaces import create_batch_analyzer, create_online_processor
from expectation_maximization import run_em_coordinate_ascent


class TestGroundTruthValidation:
    """Test state splitting against known theoretical results."""
    
    @pytest.fixture
    def test_sequences(self):
        """Generate test sequences for known processes."""
        sequences = {}
        for process_name, generator in PROCESS_GENERATORS.items():
            sequences[process_name] = generator(5000, seed=42)
        return sequences
    
    @pytest.fixture
    def empirical_data(self, test_sequences):
        """Extract empirical probabilities for all test processes."""
        empirical = {}
        for process_name, sequence in test_sequences.items():
            past_words, past_probs, future_conditional = extract_empirical_future_given_past(
                sequence, past_window_length=6, future_window_length=2
            )
            empirical[process_name] = {
                'past_words': past_words,
                'past_probs': past_probs,
                'future_conditional': future_conditional
            }
        return empirical
    
    def test_golden_mean_discovers_correct_states(self, empirical_data):
        """Golden-Mean should discover ~2 states (theoretical optimum)."""
        data = empirical_data["Golden-Mean"]
        
        # Run adaptive state growth
        beta_schedule = np.geomspace(0.5, 50, 15)
        results = run_adaptive_state_growth(
            data['past_probs'],
            data['future_conditional'],
            beta_schedule,
            maximum_states_allowed=8,
            random_seed=42
        )
        
        final_states = results['num_states'][-1]
        # Golden-Mean has finite memory, should converge to ~2 states
        assert 1 <= final_states <= 3, f"Golden-Mean discovered {final_states} states (theoretical: 2)"
        
        # Verify states are stable at final β
        stability = analyze_state_stability(
            data['past_probs'],
            results['final_posterior'],
            results['final_emission'],
            data['future_conditional'],
            beta_schedule[-1]
        )
        assert not stability['any_unstable'], "Final states should be stable"
        
        # Should converge (not keep growing)
        state_progression = results['num_states']
        final_growth = state_progression[-3:]  # Last 3 steps
        assert len(set(final_growth)) <= 2, "Should converge, not keep growing indefinitely"
    
    def test_even_process_single_state(self, empirical_data):
        """Even process should discover exactly 1 state (memoryless/IID)."""
        data = empirical_data["Even"]
        
        beta_schedule = np.geomspace(0.5, 50, 15)
        results = run_adaptive_state_growth(
            data['past_probs'],
            data['future_conditional'],
            beta_schedule,
            maximum_states_allowed=8,
            random_seed=42
        )
        
        final_states = results['num_states'][-1]
        # Even process is IID/memoryless, theoretical optimum is 1 state
        assert final_states == 1, f"Even process discovered {final_states} states (theoretical: 1 for IID)"
    
    def test_periodic_3_discovers_cycle_states(self, empirical_data):
        """Periodic-3 should discover ~3 states (matching cycle length)."""
        data = empirical_data["Periodic-3"]
        
        beta_schedule = np.geomspace(0.5, 50, 15)
        results = run_adaptive_state_growth(
            data['past_probs'],
            data['future_conditional'],
            beta_schedule,
            maximum_states_allowed=8,
            random_seed=42
        )
        
        final_states = results['num_states'][-1]
        # Periodic-3 has 3-cycle, theoretical optimum is 3 states
        assert 2 <= final_states <= 4, f"Periodic-3 discovered {final_states} states (theoretical: 3 for cycle)"
        
        # Should converge (finite cycle)
        state_progression = results['num_states']
        final_growth = state_progression[-3:]
        assert len(set(final_growth)) <= 2, "Periodic process should converge"
    
    def test_adaptive_state_growth_properties(self, empirical_data):
        """Test that adaptive state growth has correct properties."""
        data = empirical_data["Golden-Mean"]
        
        beta_schedule = np.geomspace(0.5, 50, 15)
        results = run_adaptive_state_growth(
            data['past_probs'],
            data['future_conditional'],
            beta_schedule,
            maximum_states_allowed=8,
            random_seed=42
        )
        
        # Should show monotonic or stable state count progression
        state_counts = results['num_states']
        for i in range(len(state_counts)-1):
            assert state_counts[i] <= state_counts[i+1], \
                f"State count should be monotonic: {state_counts[i]} -> {state_counts[i+1]} at step {i}"
        
        # Should respect maximum states constraint
        assert max(state_counts) <= 8, f"Should not exceed max states: {max(state_counts)}"
        
        # Should start with at least 1 state
        assert min(state_counts) >= 1, f"Should always have ≥1 state: {min(state_counts)}"
    
    def test_thue_morse_unbounded_growth(self, empirical_data):
        """Thue-Morse should show unbounded state growth (infinite statistical complexity)."""
        data = empirical_data["Thue-Morse"]
        
        # Run with higher max states to see growth pattern
        beta_schedule = np.geomspace(0.5, 50, 15)
        results = run_adaptive_state_growth(
            data['past_probs'],
            data['future_conditional'],
            beta_schedule,
            maximum_states_allowed=12,  # Higher limit
            random_seed=42
        )
        
        final_states = results['num_states'][-1]
        state_progression = results['num_states']
        
        # Should use many states (unbounded complexity)
        assert final_states >= 3, f"Thue-Morse should discover many states: {final_states}"
        
        # Should show continued growth (not converge quickly)
        growth_in_last_half = state_progression[-len(state_progression)//2:]
        initial_states = growth_in_last_half[0] 
        final_states_half = growth_in_last_half[-1]
        
        # Should continue growing or hit limit
        assert final_states_half >= initial_states, "Thue-Morse should show continued growth"
        
        # If hit limit, should be at the limit
        if final_states == 12:
            print("Thue-Morse hit state limit (expected for infinite complexity)")
        else:
            # Should be growing towards limit
            assert final_states >= 6, f"Thue-Morse should use many states: {final_states}"


class TestStructuralValidation:
    """Validate that discovered states have correct structural properties."""
    
    def test_golden_mean_emission_structure(self):
        """Test that Golden-Mean states have correct emission patterns."""
        # Generate Golden-Mean sequence
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(5000, seed=42)
        
        # Extract empirical data
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        # Run state discovery
        beta_schedule = np.geomspace(1.0, 30, 10)
        results = run_adaptive_state_growth(
            past_probs,
            future_conditional,
            beta_schedule,
            maximum_states_allowed=4,
            random_seed=42
        )
        
        emission_probs = results['final_emission']
        
        if emission_probs.shape[0] == 2:
            # One state should strongly prefer 0, other should have mixed emissions
            state_0_prob = emission_probs[0, 0]  # P(0|state_0)
            state_1_prob = emission_probs[1, 0]  # P(0|state_1)
            
            # One state should strongly favor 0, the other should be more mixed
            assert abs(state_0_prob - state_1_prob) > 0.2, \
                f"States should have different emission patterns: {state_0_prob:.3f} vs {state_1_prob:.3f}"
    
    def test_emission_probabilities_are_valid(self):
        """Test that all emission probabilities are valid probability distributions."""
        for process_name, generator in PROCESS_GENERATORS.items():
            sequence = generator(3000, seed=42)
            past_words, past_probs, future_conditional = extract_empirical_future_given_past(
                sequence, past_window_length=4, future_window_length=1
            )
            
            beta_schedule = np.geomspace(1.0, 20, 8)
            results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=6,
                random_seed=42
            )
            
            emission_probs = results['final_emission']
            
            # Check each state's emission probabilities sum to 1
            for state_idx in range(emission_probs.shape[0]):
                row_sum = emission_probs[state_idx].sum()
                assert abs(row_sum - 1.0) < 1e-6, \
                    f"{process_name} state {state_idx} emissions don't sum to 1: {row_sum}"
                
                # Check all probabilities are non-negative
                assert np.all(emission_probs[state_idx] >= 0), \
                    f"{process_name} state {state_idx} has negative probabilities"
    
    def test_posterior_distributions_are_valid(self):
        """Test that posterior distributions are valid probability distributions."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(3000, seed=42)
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        beta_schedule = np.geomspace(1.0, 20, 8)
        results = run_adaptive_state_growth(
            past_probs,
            future_conditional,
            beta_schedule,
            maximum_states_allowed=6,
            random_seed=42
        )
        
        posterior = results['final_posterior']
        
        # Check each context's posterior sums to 1
        for context_idx in range(posterior.shape[0]):
            row_sum = posterior[context_idx].sum()
            assert abs(row_sum - 1.0) < 1e-6, \
                f"Context {context_idx} posterior doesn't sum to 1: {row_sum}"
            
            # Check all probabilities are non-negative
            assert np.all(posterior[context_idx] >= 0), \
                f"Context {context_idx} has negative posterior probabilities"


class TestCrossMethodValidation:
    """Test consistency between different methods and parameters."""
    
    def test_batch_vs_online_consistency(self):
        """Test that batch and online methods discover similar state counts."""
        # Generate test sequence
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(3000, seed=42)
        
        # Batch analysis
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        beta_schedule = np.geomspace(1.0, 30, 10)
        batch_results = run_adaptive_state_growth(
            past_probs,
            future_conditional,
            beta_schedule,
            maximum_states_allowed=6,
            random_seed=42
        )
        
        # Online analysis
        processor = create_online_processor()
        for symbol in sequence:
            processor.process_symbol(symbol)
        
        online_results = processor.get_analysis_results()
        
        batch_states = batch_results['num_states'][-1]
        online_states = online_results['current_num_states']
        
        # Should be within 1 state of each other
        assert abs(batch_states - online_states) <= 1, \
            f"Batch ({batch_states}) and online ({online_states}) state counts differ too much"
    
    def test_random_seed_stability(self):
        """Test that results are consistent across different random seeds."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(3000, seed=42)
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        beta_schedule = np.geomspace(1.0, 30, 10)
        
        state_counts = []
        for seed in [42, 123, 456, 789]:
            results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=6,
                random_seed=seed
            )
            state_counts.append(results['num_states'][-1])
        
        # All runs should discover the same number of states
        unique_counts = set(state_counts)
        assert len(unique_counts) == 1, \
            f"Different seeds gave different state counts: {state_counts}"
    
    def test_beta_schedule_robustness(self):
        """Test that different β schedules converge to same result."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(3000, seed=42)
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        # Test different β schedules
        schedules = [
            np.geomspace(0.5, 50, 15),
            np.geomspace(1.0, 30, 10),
            np.geomspace(2.0, 40, 12)
        ]
        
        state_counts = []
        for schedule in schedules:
            results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                schedule,
                maximum_states_allowed=6,
                random_seed=42
            )
            state_counts.append(results['num_states'][-1])
        
        # Should converge to same state count
        unique_counts = set(state_counts)
        assert len(unique_counts) <= 2, \
            f"Different β schedules gave too different results: {state_counts}"


class TestRobustnessValidation:
    """Test robustness to various parameter changes and data conditions."""
    
    def test_sequence_length_stability(self):
        """Test that longer sequences don't change final state count."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        
        state_counts = []
        for length in [2000, 4000, 6000]:
            sequence = generator(length, seed=42)
            past_words, past_probs, future_conditional = extract_empirical_future_given_past(
                sequence, past_window_length=4, future_window_length=1
            )
            
            beta_schedule = np.geomspace(1.0, 30, 10)
            results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=6,
                random_seed=42
            )
            state_counts.append(results['num_states'][-1])
        
        # Longer sequences should give same or more stable results
        assert state_counts[-1] == state_counts[0], \
            f"State count changed with sequence length: {state_counts}"
    
    def test_window_length_effects(self):
        """Test effect of different window lengths on state discovery."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(4000, seed=42)
        
        state_counts = []
        for past_length in [3, 4, 5, 6]:
            past_words, past_probs, future_conditional = extract_empirical_future_given_past(
                sequence, past_window_length=past_length, future_window_length=1
            )
            
            beta_schedule = np.geomspace(1.0, 30, 10)
            results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=6,
                random_seed=42
            )
            state_counts.append(results['num_states'][-1])
        
        # Should be relatively stable across window lengths
        unique_counts = set(state_counts)
        assert len(unique_counts) <= 2, \
            f"Too much variation across window lengths: {state_counts}"
    
    def test_free_energy_improvement_threshold(self):
        """Test sensitivity to free energy improvement threshold."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(3000, seed=42)
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        beta_schedule = np.geomspace(1.0, 30, 10)
        
        # Test different thresholds
        thresholds = [0.001, 0.01, 0.05]
        state_counts = []
        
        for threshold in thresholds:
            results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=6,
                minimum_free_energy_improvement=threshold,
                random_seed=42
            )
            state_counts.append(results['num_states'][-1])
        
        # Higher thresholds should give same or fewer states
        assert state_counts[0] >= state_counts[1] >= state_counts[2], \
            f"State counts should decrease with higher thresholds: {state_counts}"


if __name__ == "__main__":
    # Run a quick subset of tests for development
    import pytest
    pytest.main([__file__ + "::TestGroundTruthValidation::test_golden_mean_discovers_two_states", "-v"])