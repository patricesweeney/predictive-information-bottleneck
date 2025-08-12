"""
Recursive Information Bottleneck (RIB) Implementation
===================================================

Online Recursive Information Bottleneck for discrete-discrete setting.
Converges to ε-machine predictive state partition as λ decreases.

Implementation of the three RIB fixed point equations:
1. Gibbs responsibility: p(s_t | x_t, s_{t-1}) ∝ p(s_t) exp(-1/λ KL(...))
2. Predictive model averaging: p(future | s_t) = weighted average
3. State prior: p(s_t) = marginalization
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional
import warnings


class RIB:
    """
    Recursive Information Bottleneck for online discrete-discrete learning.
    
    Implements the minimal Python API as specified:
    - RIB(alphabet_size, num_states, tau_F, alpha=1e-3, update_interval=50, seed=None)
    - partial_fit(x_t) -> s_t
    - reestimate(lambda_t, inner_iters=5)
    - objective() -> {"I_s_future": ..., "I_s_past": ..., "J": ...}
    - predictive_dist(s) -> dict[word_tuple -> prob]
    """
    
    def __init__(self, 
                 alphabet_size: int,
                 num_states: int,
                 tau_F: int,
                 alpha: float = 1e-3,
                 update_interval: int = 50,
                 seed: Optional[int] = None):
        """
        Initialize RIB with specified parameters.
        
        Args:
            alphabet_size: Size of alphabet X (finite set)
            num_states: Number of states S (fixed for a run)
            tau_F: Length of future block
            alpha: Dirichlet smoothing parameter
            update_interval: Steps between reestimate() calls
            seed: Random seed for reproducibility
        """
        self.alphabet_size = alphabet_size
        self.num_states = num_states
        self.tau_F = tau_F
        self.alpha = alpha
        self.update_interval = update_interval
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Number of possible future words
        self.num_future_words = alphabet_size ** tau_F
        
        # Initialize distributions (all in log space for numerical stability)
        self._initialize_distributions()
        
        # Ring buffer for future words
        self.ring_buffer = deque(maxlen=tau_F)
        # Queue of pending contexts to credit once future block is observed
        # Each entry: (x_t_at_context, s_{t-1}_at_context, s_t_at_context)
        self._pending_contexts = deque()
        
        # Streaming counts for p(x_t, s_{t-1})
        self.joint_counts = defaultdict(lambda: defaultdict(float))  # [x_t][s_{t-1}] -> count
        self.total_joint_count = 0.0
        
        # Empirical future word counts: p(future_word | x_t, s_{t-1})
        self.context_future_counts = defaultdict(lambda: defaultdict(float))  # [(x_t, s_{t-1})][future_word] -> count
        self.context_totals = defaultdict(float)  # (x_t, s_{t-1}) -> total_count
        
        # Current state and step tracking
        self.current_state = 0
        self.previous_state = 0
        self.step_count = 0
        
        # Track recent state assignments for active state counting
        self._recent_states = deque(maxlen=500)  # Keep last 500 assignments
        
        # Logging
        self.debug_info = {
            'min_probs': [],
            'max_probs': [],
            'normalization_errors': [],
            'nan_checks': [],
            'objective_trace': [],
            'active_states': []
        }
        
        # Optional manual temperature override used during reestimate()
        self._manual_lambda: Optional[float] = None
    
    def _initialize_distributions(self):
        """Initialize all probability distributions uniformly with smoothing."""
        # p(s_t) - state prior (in log space)
        self.log_state_prior = np.log(np.full(self.num_states, 1.0 / self.num_states))
        
        # p(future_word | s_t) - predictive model (in log space)
        # Shape: [num_states, num_future_words]
        # Initialize with slight random asymmetry to break symmetry at start
        if self.num_future_words > 1:
            random_matrix = np.random.dirichlet([1.0] * self.num_future_words, size=self.num_states)
            # Mix with uniform to keep near-uniform but distinct
            mix_weight = 0.95
            uniform_row = np.full(self.num_future_words, 1.0 / self.num_future_words)
            predictive = mix_weight * uniform_row[None, :] + (1 - mix_weight) * random_matrix
            predictive = predictive / np.sum(predictive, axis=1, keepdims=True)
            self.log_predictive_model = np.log(predictive)
        else:
            uniform_prob = 1.0
            self.log_predictive_model = np.log(np.full((self.num_states, self.num_future_words), uniform_prob))
        
        # Responsibilities p(s_t | x_t, s_{t-1}) (in log space)
        # Will be computed on demand, but initialize for tracking
        uniform_resp = 1.0 / self.num_states
        self.log_responsibilities = np.log(np.full(self.num_states, uniform_resp))
        
        # Count matrices for reestimation
        self.state_counts = np.zeros(self.num_states)
        self.predictive_counts = np.zeros((self.num_states, self.num_future_words))
        
    def partial_fit(self, x_t: int, run_log=None) -> int:
        """
        Online step: process new symbol and return current state.
        
        Args:
            x_t: Current symbol from alphabet
            run_log: Optional RIBRunLog instance for tracking metrics
            
        Returns:
            s_t: Current state assignment
        """
        # Add current symbol to ring buffer (represents latest symbol of any future block)
        self.ring_buffer.append(x_t)
        
        # Update step count
        self.step_count += 1
        
        # Compute responsibilities p(s_t | x_t, s_{t-1})
        self.log_responsibilities = self._compute_gibbs_responsibilities(x_t, self.previous_state)
        
        # Choose current state (argmax for deterministic, or sample)
        s_t = self._choose_state()
        
        # Update streaming counts for p(x_t, s_{t-1})
        self.joint_counts[x_t][self.previous_state] += 1.0
        self.total_joint_count += 1.0
        
        # Queue this context for delayed crediting once its future block is available
        # We credit the context from tau_F steps ago with the current ring-buffer future block
        self._pending_contexts.append((x_t, self.previous_state, s_t))
        if len(self._pending_contexts) > self.tau_F and len(self.ring_buffer) == self.tau_F:
            # Oldest pending context now has its future block observed in the ring buffer
            context_x_t, context_s_prev, context_s_t = self._pending_contexts.popleft()
            future_word = self._ring_buffer_to_word_index()
            self._credit_future_word(future_word, context_x_t, context_s_prev, context_s_t)
        
        # Update state tracking
        self.previous_state = self.current_state
        self.current_state = s_t
        
        # Track recent states for active counting
        self._recent_states.append(s_t)
        
        # Periodic reestimation and logging
        if self.step_count % self.update_interval == 0:
            # Get current lambda from deterministic annealing schedule
            lambda_t = self._get_current_lambda()
            self.reestimate(lambda_t)
            
            # Log metrics if run_log provided
            if run_log is not None:
                objective = self.objective()
                run_log.log_step(self.step_count, self, lambda_t, objective)
                
                # Log detailed checkpoint every few intervals
                if self.step_count % (self.update_interval * 10) == 0:
                    run_log.log_checkpoint(self.step_count, self, lambda_t, 
                                         f"step_{self.step_count}")
        
        return s_t
    
    def _compute_gibbs_responsibilities(self, x_t: int, s_prev: int) -> np.ndarray:
        """
        Compute Gibbs responsibilities: p(s_t | x_t, s_{t-1}) ∝ p(s_t) exp(-1/λ KL(...))
        
        Returns log responsibilities.
        """
        lambda_t = self._get_current_lambda()
        # Clip lambda away from 0 to avoid infinite sharpness
        lambda_t = max(lambda_t, 1e-3)
        
        # Get empirical future distribution p(future | x_t, s_{t-1})
        log_empirical_future = self._get_empirical_future_dist(x_t, s_prev)
        
        log_responsibilities = np.full(self.num_states, -np.inf)
        
        for s_t in range(self.num_states):
            # Prior term: log p(s_t)
            log_prior_term = self.log_state_prior[s_t]
            
            # KL divergence term: D_KL(empirical || model)
            kl_div = self._compute_kl_divergence(log_empirical_future, self.log_predictive_model[s_t])
            
            # Gibbs weight: log p(s_t) - (1/λ) * KL
            log_responsibilities[s_t] = log_prior_term - (1.0 / lambda_t) * kl_div
        
        # Normalize using log-sum-exp
        log_responsibilities = self._log_normalize(log_responsibilities)
        
        # Temperature sharpening regularization to encourage separation as lambda lowers
        if lambda_t < 0.1:
            sharpen = min(5.0, 0.1 / lambda_t)
            log_responsibilities = self._log_normalize(log_responsibilities * sharpen)
        
        return log_responsibilities
    
    def _get_empirical_future_dist(self, x_t: int, s_prev: int) -> np.ndarray:
        """
        Get empirical p(future | x_t, s_{t-1}) from observed data.
        
        Returns log probabilities with Dirichlet smoothing.
        """
        context = (x_t, s_prev)
        
        # Start with Dirichlet smoothing
        counts = np.full(self.num_future_words, self.alpha)
        
        # Add observed counts for this context
        if context in self.context_future_counts:
            for future_word, count in self.context_future_counts[context].items():
                if 0 <= future_word < self.num_future_words:
                    counts[future_word] += count
        
        # Normalize and return log probabilities
        total_count = np.sum(counts)
        log_probs = np.log(counts / total_count)
        
        return log_probs
    
    def _compute_kl_divergence(self, log_p: np.ndarray, log_q: np.ndarray) -> float:
        """
        Compute KL divergence D_KL(P||Q) in log space.
        
        Args:
            log_p: Log probabilities of distribution P
            log_q: Log probabilities of distribution Q
            
        Returns:
            KL divergence value
        """
        # Convert to linear space for KL computation
        p = np.exp(log_p)
        q = np.exp(log_q)
        
        # Ensure no zeros for numerical stability
        p = np.maximum(p, 1e-15)
        q = np.maximum(q, 1e-15)
        
        # Renormalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Compute KL divergence
        kl_div = np.sum(p * (np.log(p) - np.log(q)))
        
        return kl_div
    
    def _choose_state(self) -> int:
        """Choose current state from responsibilities (argmax for deterministic)."""
        # Use argmax for deterministic assignment
        responsibilities = np.exp(self.log_responsibilities)
        return int(np.argmax(responsibilities))
    
    def _ring_buffer_to_word_index(self) -> int:
        """Convert ring buffer contents to future word index."""
        # Interpret the buffer as a base-alphabet_size number with most recent symbol as least significant
        word_index = 0
        base = 1
        for symbol in reversed(self.ring_buffer):
            word_index += symbol * base
            base *= self.alphabet_size
        return word_index
    
    def _credit_future_word(self, future_word: int, x_t: int, s_prev: int, s_t: int):
        """Credit completed future word to counts."""
        # Update empirical context-future counts
        context = (x_t, s_prev)
        self.context_future_counts[context][future_word] += 1.0
        self.context_totals[context] += 1.0
        
        # Update count matrices for reestimation
        self.state_counts[s_t] += 1.0
        self.predictive_counts[s_t, future_word] += 1.0
    
    def _get_current_lambda(self) -> float:
        """Get current lambda from deterministic annealing schedule."""
        # During reestimate() we respect a manual override for consistent inner updates
        if self._manual_lambda is not None:
            return self._manual_lambda
        # More aggressive geometric schedule: 1.0 -> 0.005
        max_steps = 8000  # Faster annealing
        progress = min(self.step_count / max_steps, 1.0)
        
        # Two-phase annealing: slower at first, then more aggressive
        if progress < 0.5:
            # Phase 1: 1.0 -> 0.1 (slower)
            phase_progress = progress * 2
            lambda_t = 1.0 * (0.1 / 1.0) ** phase_progress
        else:
            # Phase 2: 0.1 -> 0.005 (faster)
            phase_progress = (progress - 0.5) * 2
            lambda_t = 0.1 * (0.005 / 0.1) ** phase_progress
        
        return max(lambda_t, 0.005)  # Lower floor
    
    def reestimate(self, lambda_t: float, inner_iters: int = 5):
        """
        Coordinate updates enforcing the three RIB fixed point equations.
        
        Args:
            lambda_t: Current temperature parameter
            inner_iters: Number of inner coordinate ascent iterations
        """
        # Use the provided temperature during these inner iterations
        previous_manual_lambda = self._manual_lambda
        self._manual_lambda = lambda_t
        try:
            for iteration in range(inner_iters):
                # Store objective before update
                obj_before = self.objective()
                
                # 1. Update predictive model: p(future | s_t)
                self._update_predictive_model()
                
                # 2. Update state prior: p(s_t)
                self._update_state_prior()
                
                # 3. Recompute responsibilities (done in partial_fit)
                
                # Check objective improvement
                obj_after = self.objective()
                
                # Log objective trace
                self.debug_info['objective_trace'].append(obj_after['J'])
                
                # Check for monotonic increase (with tolerance)
                if obj_after['J'] < obj_before['J'] - 1e-6:
                    warnings.warn(f"Objective decreased in iteration {iteration}: {obj_before['J']:.6f} -> {obj_after['J']:.6f}")
        finally:
            # Restore prior override state
            self._manual_lambda = previous_manual_lambda
            
            # Update active states count
            active_states = self._count_active_states()
            self.debug_info['active_states'].append(active_states)
    
    def _update_predictive_model(self):
        """
        Update p(future | s_t) using predictive model averaging:
        p(future | s_t) = Σ_{x_t, s_{t-1}} p(future | x_t, s_{t-1}) * p(s_t | x_t, s_{t-1}) * p(x_t, s_{t-1}) / p(s_t)
        """
        lambda_t = self._get_current_lambda()
        for s_t in range(self.num_states):
            # Get normalization factor p(s_t)
            state_prior = np.exp(self.log_state_prior[s_t])
            
            if state_prior < 1e-15:
                continue  # Skip inactive states
            
            # Accumulate weighted predictions with Dirichlet smoothing in both numerator and denominator
            weighted_predictions = np.full(self.num_future_words, self.alpha)
            total_weight = self.alpha * self.num_future_words
            
            # Iterate over observed (x_t, s_{t-1}) pairs
            for x_t in self.joint_counts:
                for s_prev in self.joint_counts[x_t]:
                    joint_prob = self.joint_counts[x_t][s_prev] / max(self.total_joint_count, 1e-15)
                    
                    # Get responsibility p(s_t | x_t, s_{t-1})
                    log_resp = self._compute_gibbs_responsibilities(x_t, s_prev)
                    responsibility = np.exp(log_resp[s_t])
                    
                    # Weight for this context
                    weight = responsibility * joint_prob / state_prior
                    
                    # Get empirical future distribution
                    log_empirical = self._get_empirical_future_dist(x_t, s_prev)
                    empirical = np.exp(log_empirical)
                    
                    # Add weighted contribution
                    weighted_predictions += weight * empirical
                    total_weight += weight
            
            # Normalize and update in log space
            if total_weight > 0:
                self.log_predictive_model[s_t] = np.log(weighted_predictions / total_weight)
            else:
                # Fallback to uniform if no observations
                self.log_predictive_model[s_t] = np.log(np.full(self.num_future_words, 1.0 / self.num_future_words))
        
        # Symmetry breaking at low temperature to enable state differentiation
        if lambda_t < 0.2 and self.num_future_words > 1:
            for s_t in range(self.num_states):
                state_prior = np.exp(self.log_state_prior[s_t])
                if state_prior < 1e-9:
                    continue
                probs = np.exp(self.log_predictive_model[s_t])
                if np.std(probs) < 1e-6:
                    noise = np.random.dirichlet([1.0] * self.num_future_words) * 1e-3
                    perturbed = probs + noise
                    perturbed = perturbed / np.sum(perturbed)
                    self.log_predictive_model[s_t] = np.log(perturbed)
        
        # Sanity check: ensure distributions are normalized
        self._check_normalization()
    
    def _update_state_prior(self):
        """
        Update p(s_t) = Σ_{x_t, s_{t-1}} p(s_t | x_t, s_{t-1}) * p(x_t, s_{t-1})
        """
        new_state_prior = np.full(self.num_states, self.alpha)  # Smoothing
        
        # If we have no observations yet, keep uniform
        if self.total_joint_count < 1e-15:
            return
        
        # Iterate over observed (x_t, s_{t-1}) pairs
        for x_t in self.joint_counts:
            for s_prev in self.joint_counts[x_t]:
                joint_prob = self.joint_counts[x_t][s_prev] / self.total_joint_count
                
                # Get responsibilities for this context
                log_resp = self._compute_gibbs_responsibilities(x_t, s_prev)
                responsibilities = np.exp(log_resp)
                
                # Add contribution to each state (weighted by joint probability)
                new_state_prior += responsibilities * joint_prob * self.total_joint_count
        
        # Normalize and update in log space
        total_prior = np.sum(new_state_prior)
        if total_prior > 1e-15:
            self.log_state_prior = np.log(new_state_prior / total_prior)
        
        # Apply temperature-based sharpening to encourage proper convergence
        lambda_t = self._get_current_lambda()
        if lambda_t < 0.1:
            # At low temperature, sharpen the distribution more aggressively
            sharpening_factor = 1.0 / max(lambda_t, 0.01)
            self.log_state_prior = self.log_state_prior * sharpening_factor
            self.log_state_prior = self._log_normalize(self.log_state_prior)
            
            # Also apply entropy regularization to further collapse unused states
            state_priors = np.exp(self.log_state_prior)
            min_usage = 1.0 / self.num_states * 0.001  # Very small threshold
            
            # Zero out states that are barely used
            state_priors[state_priors < min_usage] = min_usage / 10
            self.log_state_prior = np.log(state_priors / np.sum(state_priors))
    
    def objective(self) -> Dict[str, float]:
        """
        Compute objective J(λ) = I(s_t; future) - λ * I(s_t; s_{t-1}, x_t).
        
        Returns:
            Dictionary with I_s_future, I_s_past, and J values
        """
        # Compute mutual informations
        I_s_future = self._compute_mutual_information_s_future()
        I_s_past = self._compute_mutual_information_s_past()
        
        lambda_t = self._get_current_lambda()
        # Variational objective per Still: maximize I(S;F) - λ I(S;Past)
        J = I_s_future - lambda_t * I_s_past
        
        return {
            'I_s_future': I_s_future,
            'I_s_past': I_s_past,
            'J': J
        }
    
    def _compute_mutual_information_s_future(self) -> float:
        """Compute I(s_t; future) using current distributions."""
        # Simplified MI calculation
        # I(S;F) = Σ p(s,f) log(p(s,f) / (p(s)p(f)))
        
        mi = 0.0
        state_priors = np.exp(self.log_state_prior)
        
        # Estimate future marginal
        future_marginal = np.zeros(self.num_future_words)
        for s in range(self.num_states):
            future_given_s = np.exp(self.log_predictive_model[s])
            future_marginal += state_priors[s] * future_given_s
        
        # Compute MI
        for s in range(self.num_states):
            if state_priors[s] < 1e-15:
                continue
            
            future_given_s = np.exp(self.log_predictive_model[s])
            for f in range(self.num_future_words):
                if future_given_s[f] < 1e-15 or future_marginal[f] < 1e-15:
                    continue
                
                joint_prob = state_priors[s] * future_given_s[f]
                mi += joint_prob * np.log(joint_prob / (state_priors[s] * future_marginal[f]))
        
        # Numerical floor for non-negativity
        return max(float(mi), 0.0)
    
    def _compute_mutual_information_s_past(self) -> float:
        """Compute I(s_t; (s_{t-1}, x_t)) using current encoder and empirical context distribution."""
        if self.total_joint_count < 1e-15:
            return 0.0
        
        epsilon = 1e-15
        state_priors = np.clip(np.exp(self.log_state_prior), epsilon, 1.0)
        log_state_priors = np.log(state_priors)
        
        mi = 0.0
        for x_t in self.joint_counts:
            for s_prev, count in self.joint_counts[x_t].items():
                p_context = count / self.total_joint_count
                # Current responsibilities q(s|context)
                log_resp = self._compute_gibbs_responsibilities(x_t, s_prev)
                q_s_given_context = np.clip(np.exp(log_resp), epsilon, 1.0)
                log_q = np.log(q_s_given_context)
                mi += p_context * np.sum(q_s_given_context * (log_q - log_state_priors))
        # Numerical floor for non-negativity
        return max(float(mi), 0.0)
    
    def predictive_dist(self, s: int) -> Dict[Tuple[int, ...], float]:
        """
        Return p(future | s) for inspection and testing.
        
        Args:
            s: State index
            
        Returns:
            Dictionary mapping future word tuples to probabilities
        """
        if s >= self.num_states:
            raise ValueError(f"State {s} >= num_states {self.num_states}")
        
        result = {}
        probs = np.exp(self.log_predictive_model[s])
        
        for word_idx in range(self.num_future_words):
            # Convert word index back to tuple
            word_tuple = self._word_index_to_tuple(word_idx)
            result[word_tuple] = probs[word_idx]
        
        return result
    
    def _word_index_to_tuple(self, word_idx: int) -> Tuple[int, ...]:
        """Convert word index back to tuple of symbols."""
        symbols = []
        remaining = word_idx
        for _ in range(self.tau_F):
            symbols.append(remaining % self.alphabet_size)
            remaining //= self.alphabet_size
        return tuple(symbols)
    
    def _log_normalize(self, log_probs: np.ndarray) -> np.ndarray:
        """Normalize log probabilities using log-sum-exp."""
        max_log_prob = np.max(log_probs)
        if not np.isfinite(max_log_prob):
            # All probabilities are -inf, return uniform
            return np.log(np.full(len(log_probs), 1.0 / len(log_probs)))
        
        shifted_probs = log_probs - max_log_prob
        log_sum = max_log_prob + np.log(np.sum(np.exp(shifted_probs)))
        
        return log_probs - log_sum
    
    def _check_normalization(self):
        """Check that all distributions are properly normalized."""
        # Check state prior
        state_prior_sum = np.sum(np.exp(self.log_state_prior))
        if abs(state_prior_sum - 1.0) > 1e-8:
            self.debug_info['normalization_errors'].append(f"State prior sum: {state_prior_sum}")
        
        # Check predictive models
        for s in range(self.num_states):
            pred_sum = np.sum(np.exp(self.log_predictive_model[s]))
            if abs(pred_sum - 1.0) > 1e-8:
                self.debug_info['normalization_errors'].append(f"Predictive model {s} sum: {pred_sum}")
        
        # Check for NaNs
        if np.any(~np.isfinite(self.log_state_prior)) or np.any(~np.isfinite(self.log_predictive_model)):
            self.debug_info['nan_checks'].append(f"NaN detected at step {self.step_count}")
    
    def _count_active_states(self) -> int:
        """Count number of truly active states based on recent usage."""
        # Count states actually used in recent assignments
        recent_window = 200  # Look at last 200 assignments
        if hasattr(self, '_recent_states'):
            # Use cached recent states for efficiency
            recent_states = self._recent_states
        else:
            # Fallback to prior-based counting
            state_priors = np.exp(self.log_state_prior)
            threshold = 1.0 / self.num_states * 0.01
            return int(np.sum(state_priors > threshold))
        
        if len(recent_states) < 10:
            # Not enough data, fall back to prior-based count
            state_priors = np.exp(self.log_state_prior)
            threshold = 1.0 / self.num_states * 0.01
            return int(np.sum(state_priors > threshold))
        
        # Count unique states in recent window
        # Convert deque to list for slicing
        recent_list = list(recent_states)
        recent_subset = recent_list[-recent_window:] if len(recent_list) >= recent_window else recent_list
        unique_states = set(recent_subset)
        return len(unique_states)
    
    def get_debug_info(self) -> Dict:
        """Get debugging information."""
        return self.debug_info.copy()