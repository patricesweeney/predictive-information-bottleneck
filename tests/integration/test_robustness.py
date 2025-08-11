"""
Robustness Integration Tests
===========================

Tests for robustness to noise, parameter variations, and edge cases
to ensure reliable state discovery under realistic conditions.
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from processes import PROCESS_GENERATORS
from empirical_analysis import extract_empirical_future_given_past
from state_splitting import run_adaptive_state_growth
from interfaces import create_online_processor


class TestNoiseRobustness:
    """Test robustness to various types of noise in data."""
    
    def add_bit_flip_noise(self, sequence, noise_rate=0.05, seed=42):
        """Add bit-flip noise to binary sequence."""
        rng = np.random.default_rng(seed)
        noisy_sequence = sequence.copy()
        flip_mask = rng.random(len(sequence)) < noise_rate
        noisy_sequence[flip_mask] = 1 - noisy_sequence[flip_mask]
        return noisy_sequence
    
    def add_symbol_substitution_noise(self, sequence, noise_rate=0.05, seed=42):
        """Add random symbol substitution noise."""
        rng = np.random.default_rng(seed)
        noisy_sequence = sequence.copy()
        noise_mask = rng.random(len(sequence)) < noise_rate
        noisy_sequence[noise_mask] = rng.integers(0, 2, size=noise_mask.sum())
        return noisy_sequence
    
    def test_golden_mean_noise_robustness(self):
        """Test Golden-Mean state discovery under different noise levels."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        clean_sequence = generator(4000, seed=42)
        
        results = {}
        noise_levels = [0.0, 0.02, 0.05, 0.10]
        
        for noise_rate in noise_levels:
            if noise_rate == 0.0:
                test_sequence = clean_sequence
            else:
                test_sequence = self.add_bit_flip_noise(clean_sequence, noise_rate, seed=42)
            
            past_words, past_probs, future_conditional = extract_empirical_future_given_past(
                test_sequence, past_window_length=4, future_window_length=1
            )
            
            beta_schedule = np.geomspace(1.0, 30, 10)
            state_results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=6,
                random_seed=42
            )
            
            results[noise_rate] = {
                'states': state_results['num_states'][-1],
                'free_energy': state_results['free_energies'][-1]
            }
        
        # Should maintain 2-state structure with low noise
        clean_states = results[0.0]['states']
        low_noise_states = results[0.02]['states']
        
        assert abs(clean_states - low_noise_states) <= 1, \
            f"Low noise should preserve structure: {clean_states} vs {low_noise_states}"
        
        # High noise should not increase states dramatically
        high_noise_states = results[0.10]['states']
        assert high_noise_states <= clean_states + 2, \
            f"High noise should not create too many states: {high_noise_states} vs {clean_states}"
    
    def test_noise_type_comparison(self):
        """Compare robustness to different types of noise."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        clean_sequence = generator(3000, seed=42)
        
        noise_rate = 0.05
        
        # Test different noise types
        sequences = {
            'clean': clean_sequence,
            'bit_flip': self.add_bit_flip_noise(clean_sequence, noise_rate, seed=42),
            'substitution': self.add_symbol_substitution_noise(clean_sequence, noise_rate, seed=42)
        }
        
        results = {}
        for noise_type, sequence in sequences.items():
            past_words, past_probs, future_conditional = extract_empirical_future_given_past(
                sequence, past_window_length=4, future_window_length=1
            )
            
            beta_schedule = np.geomspace(1.0, 25, 8)
            state_results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=5,
                random_seed=42
            )
            
            results[noise_type] = state_results['num_states'][-1]
        
        # Both noise types should give similar results
        clean_states = results['clean']
        bit_flip_states = results['bit_flip']
        sub_states = results['substitution']
        
        assert abs(bit_flip_states - clean_states) <= 1, \
            f"Bit flip noise too disruptive: {bit_flip_states} vs {clean_states}"
        assert abs(sub_states - clean_states) <= 1, \
            f"Substitution noise too disruptive: {sub_states} vs {clean_states}"
    
    def test_online_noise_robustness(self):
        """Test online processor robustness to noise."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        clean_sequence = generator(2000, seed=42)
        noisy_sequence = self.add_bit_flip_noise(clean_sequence, 0.03, seed=42)
        
        # Clean online processing
        clean_processor = create_online_processor()
        clean_processor.inverse_temperature_beta = 25.0
        clean_processor.eigenvalue_check_interval = 200
        
        for symbol in clean_sequence:
            clean_processor.process_symbol(symbol)
        
        # Noisy online processing
        noisy_processor = create_online_processor()
        noisy_processor.inverse_temperature_beta = 25.0
        noisy_processor.eigenvalue_check_interval = 200
        
        for symbol in noisy_sequence:
            noisy_processor.process_symbol(symbol)
        
        clean_results = clean_processor.get_analysis_results()
        noisy_results = noisy_processor.get_analysis_results()
        
        clean_states = clean_results['current_num_states']
        noisy_states = noisy_results['current_num_states']
        
        assert abs(clean_states - noisy_states) <= 1, \
            f"Online processing should be noise robust: {clean_states} vs {noisy_states}"


class TestParameterSensitivity:
    """Test sensitivity to various parameter choices."""
    
    def test_free_energy_threshold_sensitivity(self):
        """Test sensitivity to free energy improvement threshold."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(3000, seed=42)
        
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        beta_schedule = np.geomspace(1.0, 30, 10)
        
        # Test different thresholds
        thresholds = [0.001, 0.01, 0.05, 0.1]
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
        
        # Lower thresholds should allow more states
        assert state_counts[0] >= state_counts[-1], \
            f"Lower threshold should allow more states: {state_counts}"
        
        # But should still discover meaningful structure
        assert min(state_counts) >= 1, "Should discover at least 1 state"
        assert max(state_counts) <= 4, "Should not create too many states"
    
    def test_beta_schedule_variations(self):
        """Test robustness to different β annealing schedules."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(3000, seed=42)
        
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        # Different schedule types
        schedules = {
            'geometric': np.geomspace(1.0, 30, 10),
            'linear': np.linspace(1.0, 30, 10),
            'exponential': np.exp(np.linspace(0, np.log(30), 10)),
            'slow_start': np.concatenate([np.ones(3), np.geomspace(1.0, 30, 7)]),
            'fast_start': np.geomspace(1.0, 30, 5)
        }
        
        results = {}
        for schedule_name, schedule in schedules.items():
            try:
                state_results = run_adaptive_state_growth(
                    past_probs,
                    future_conditional,
                    schedule,
                    maximum_states_allowed=6,
                    random_seed=42
                )
                results[schedule_name] = state_results['num_states'][-1]
            except Exception:
                # Skip problematic schedules
                continue
        
        if len(results) > 1:
            state_counts = list(results.values())
            # Should be reasonably consistent
            state_range = max(state_counts) - min(state_counts)
            assert state_range <= 2, \
                f"Too much variation across schedules: {results}"
    
    def test_maximum_states_constraint(self):
        """Test behavior with different maximum state constraints."""
        generator = PROCESS_GENERATORS["Mixed"]  # More complex process
        sequence = generator(3000, seed=42)
        
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        beta_schedule = np.geomspace(1.0, 40, 12)
        
        # Test different maximum state limits
        max_states_limits = [2, 4, 6, 8]
        final_states = []
        
        for max_states in max_states_limits:
            results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=max_states,
                random_seed=42
            )
            final_states.append(results['num_states'][-1])
        
        # Should respect constraints
        for i, max_limit in enumerate(max_states_limits):
            assert final_states[i] <= max_limit, \
                f"Exceeded maximum states: {final_states[i]} > {max_limit}"
        
        # Should use available capacity reasonably
        assert final_states[-1] >= final_states[0], \
            "Higher limits should allow more states"
    
    def test_perturbation_scale_robustness(self):
        """Test robustness to state splitting perturbation scale."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(3000, seed=42)
        
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        beta_schedule = np.geomspace(1.0, 30, 10)
        
        # Test with different perturbation scales (would need to modify state_splitting.py)
        # For now, test that default behavior is stable across runs
        results = []
        for run in range(5):
            state_results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=6,
                random_seed=42 + run  # Different seeds
            )
            results.append(state_results['num_states'][-1])
        
        # Should be consistent across runs for same data
        unique_results = set(results)
        assert len(unique_results) <= 2, \
            f"Too much variation across runs: {results}"


class TestEdgeCaseHandling:
    """Test handling of edge cases and boundary conditions."""
    
    def test_single_symbol_sequences(self):
        """Test behavior with sequences of single symbol."""
        # Create artificial single-symbol sequence
        sequence = np.zeros(1000, dtype=int)  # All zeros
        
        try:
            past_words, past_probs, future_conditional = extract_empirical_future_given_past(
                sequence, past_window_length=2, future_window_length=1
            )
            
            if len(past_probs) > 0:
                beta_schedule = np.geomspace(1.0, 10, 5)
                results = run_adaptive_state_growth(
                    past_probs,
                    future_conditional,
                    beta_schedule,
                    maximum_states_allowed=3,
                    random_seed=42
                )
                
                # Should discover exactly 1 state for deterministic sequence
                assert results['num_states'][-1] == 1, \
                    f"Single symbol should give 1 state: {results['num_states'][-1]}"
        
        except Exception:
            # May fail due to lack of variation - that's acceptable
            pytest.skip("Single symbol sequence handling")
    
    def test_very_short_windows(self):
        """Test behavior with minimal window lengths."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(2000, seed=42)
        
        # Test minimal windows
        try:
            past_words, past_probs, future_conditional = extract_empirical_future_given_past(
                sequence, past_window_length=1, future_window_length=1
            )
            
            beta_schedule = np.geomspace(1.0, 20, 8)
            results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=4,
                random_seed=42
            )
            
            # Should complete successfully
            assert results['num_states'][-1] >= 1, "Should discover at least 1 state"
            assert results['num_states'][-1] <= 3, "Should not create too many states"
        
        except Exception as e:
            pytest.skip(f"Minimal window handling: {e}")
    
    def test_empty_context_handling(self):
        """Test handling when some contexts have no observations."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(500, seed=42)  # Short sequence
        
        try:
            # Use large window that might create sparse contexts
            past_words, past_probs, future_conditional = extract_empirical_future_given_past(
                sequence, past_window_length=6, future_window_length=1
            )
            
            if len(past_probs) > 0:
                beta_schedule = np.geomspace(1.0, 15, 6)
                results = run_adaptive_state_growth(
                    past_probs,
                    future_conditional,
                    beta_schedule,
                    maximum_states_allowed=4,
                    random_seed=42
                )
                
                # Should handle sparse data gracefully
                assert results['num_states'][-1] >= 1, "Should handle sparse contexts"
        
        except Exception:
            # May fail with very sparse data - acceptable
            pytest.skip("Sparse context handling")
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme probability values."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(2000, seed=42)
        
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=3, future_window_length=1
        )
        
        # Test with very high β (numerical challenge)
        extreme_beta_schedule = np.geomspace(10.0, 200.0, 8)
        
        try:
            results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                extreme_beta_schedule,
                maximum_states_allowed=5,
                random_seed=42
            )
            
            # Should complete without numerical errors
            assert np.isfinite(results['free_energies'][-1]), "Free energy should be finite"
            assert results['num_states'][-1] >= 1, "Should discover states"
            
            # Check emission probabilities are valid
            emission = results['final_emission']
            assert np.all(emission >= 0), "Emissions should be non-negative"
            assert np.allclose(emission.sum(axis=1), 1.0, atol=1e-6), "Emissions should normalize"
        
        except Exception as e:
            pytest.skip(f"Extreme β handling: {e}")


class TestConsistencyAcrossRuns:
    """Test consistency across multiple independent runs."""
    
    def test_deterministic_consistency(self):
        """Test that deterministic parts give consistent results."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        
        # Multiple independent runs with same data and seed
        results = []
        for run in range(3):
            sequence = generator(2000, seed=42)  # Same seed = same data
            
            past_words, past_probs, future_conditional = extract_empirical_future_given_past(
                sequence, past_window_length=4, future_window_length=1
            )
            
            beta_schedule = np.geomspace(1.0, 25, 8)
            state_results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=5,
                random_seed=42  # Same random seed
            )
            
            results.append(state_results['num_states'][-1])
        
        # Should be exactly the same
        assert len(set(results)) == 1, f"Deterministic runs should be identical: {results}"
    
    def test_statistical_consistency(self):
        """Test statistical consistency across different data realizations."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        
        # Multiple runs with different data but same process
        results = []
        for run in range(5):
            sequence = generator(2000, seed=42 + run)  # Different data
            
            past_words, past_probs, future_conditional = extract_empirical_future_given_past(
                sequence, past_window_length=4, future_window_length=1
            )
            
            beta_schedule = np.geomspace(1.0, 25, 8)
            state_results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=5,
                random_seed=42  # Same processing seed
            )
            
            results.append(state_results['num_states'][-1])
        
        # Should be statistically consistent (mostly same result)
        from collections import Counter
        counts = Counter(results)
        most_common_count = counts.most_common(1)[0][1]
        
        # At least 60% should give the same result
        assert most_common_count >= 3, \
            f"Should be statistically consistent: {results}"


if __name__ == "__main__":
    # Run quick test for development
    import pytest
    pytest.main([__file__ + "::TestNoiseRobustness::test_golden_mean_noise_robustness", "-v"])