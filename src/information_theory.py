"""
Information Theory and Free Energy Computations
==============================================

Core information-theoretic functions for variational free energy (VFE)
computations in the information bottleneck framework.

Includes:
- KL divergence and related measures
- Free energy computation for IB
- Eigenvalue analysis for phase transitions
"""

import numpy as np


def compute_kl_divergence(probability_p, probability_q, epsilon=1e-12):
    """
    Compute Kullback-Leibler divergence D_KL(P || Q).
    
    KL divergence measures the information lost when using Q to approximate P.
    
    Args:
        probability_p: "True" probability distribution
        probability_q: Approximating probability distribution  
        epsilon: Small constant to avoid log(0)
        
    Returns:
        float: KL divergence value (non-negative)
    """
    # Clip probabilities to avoid numerical issues
    p_clipped = np.clip(probability_p, epsilon, 1.0)
    q_clipped = np.clip(probability_q, epsilon, 1.0)
    
    return np.sum(p_clipped * np.log(p_clipped / q_clipped))


def compute_jensen_shannon_divergence(probability_p, probability_q, epsilon=1e-12):
    """
    Compute Jensen-Shannon divergence - symmetric version of KL divergence.
    
    JS divergence is the average of KL divergences from each distribution
    to their average: JS(P,Q) = 0.5 * [KL(P||M) + KL(Q||M)] where M = 0.5*(P+Q)
    
    Args:
        probability_p: First probability distribution
        probability_q: Second probability distribution
        epsilon: Small constant to avoid numerical issues
        
    Returns:
        float: Jensen-Shannon divergence (0 to 1)
    """
    # Average distribution
    mixture_distribution = 0.5 * (probability_p + probability_q)
    
    # Symmetric KL divergences
    kl_p_to_mixture = compute_kl_divergence(probability_p, mixture_distribution, epsilon)
    kl_q_to_mixture = compute_kl_divergence(probability_q, mixture_distribution, epsilon)
    
    return 0.5 * (kl_p_to_mixture + kl_q_to_mixture)


def compute_variational_free_energy(past_probabilities, 
                                  posterior_distribution, 
                                  emission_probabilities, 
                                  inverse_temperature_beta,
                                  future_conditional_probabilities):
    """
    Compute Blahut-Arimoto variational free energy with both decompositions.
    
    Following the Blahut-Arimoto formulation:
    F_β[q,p] = E_p(c)[KL(q(s|c)||p(s)) + β·E_q(s|c)KL(r(·|c)||p(·|s))]
    
    Two equivalent decompositions:
    1. Accuracy-Complexity: F = Complexity - β·Accuracy + const
    2. Energy-Entropy: F = Energy - Entropy + const
    
    Args:
        past_probabilities: p(c) - context probabilities [N_past]
        posterior_distribution: q(s|c) - encoder [N_past, N_states] 
        emission_probabilities: p(y|s) - decoder [N_states, N_future]
        inverse_temperature_beta: β - inverse temperature
        future_conditional_probabilities: r(y|c) - empirical conditional [N_past, N_future]
        
    Returns:
        tuple: (free_energy, complexity, accuracy, energy, entropy)
    """
    epsilon = 1e-12
    
    # ===============================================================
    # BLAHUT-ARIMOTO VFE: Core computation
    # ===============================================================
    
    # Compute prior p(s) = sum_c p(c) q(s|c)
    prior_distribution = np.sum(
        past_probabilities[:, None] * posterior_distribution, axis=0
    )
    prior_distribution = np.clip(prior_distribution, epsilon, 1.0)
    
    # ===============================================================
    # DECOMPOSITION 1: Accuracy-Complexity
    # ===============================================================
    
    # Complexity: E_p(c)[KL(q(s|c)||p(s))]
    # = sum_c p(c) sum_s q(s|c) log(q(s|c)/p(s))
    log_posterior = np.log(np.clip(posterior_distribution, epsilon, 1.0))
    log_prior = np.log(prior_distribution)
    
    complexity = np.sum(
        past_probabilities[:, None] * posterior_distribution * 
        (log_posterior - log_prior[None, :])
    )
    
    # Accuracy: E_p(c) E_q(s|c) E_r(y|c)[log p(y|s)]
    # = sum_c p(c) sum_s q(s|c) sum_y r(y|c) log p(y|s)
    log_emission = np.log(np.clip(emission_probabilities, epsilon, 1.0))
    
    accuracy = np.sum(
        past_probabilities[:, None, None] * 
        posterior_distribution[:, :, None] *
        future_conditional_probabilities[:, None, :] *
        log_emission[None, :, :]
    )
    
    # ===============================================================
    # DECOMPOSITION 2: Energy-Entropy  
    # ===============================================================
    
    # Energy: E_p(c) E_q(s|c)[-log p(s) - β·E_r(y|c)[log p(y|s)]]
    # = -sum_c p(c) sum_s q(s|c) [log p(s) + β·sum_y r(y|c) log p(y|s)]
    energy_term1 = -np.sum(
        past_probabilities[:, None] * posterior_distribution * log_prior[None, :]
    )
    
    energy_term2 = -inverse_temperature_beta * accuracy
    
    energy = energy_term1 + energy_term2
    
    # Entropy: E_p(c) H(q(s|c))
    # = -sum_c p(c) sum_s q(s|c) log q(s|c)
    entropy = -np.sum(
        past_probabilities[:, None] * posterior_distribution * log_posterior
    )
    
    # ===============================================================
    # FREE ENERGY (both decompositions must be equal)
    # ===============================================================
    
    # Decomposition 1: F = Complexity - β·Accuracy
    free_energy_1 = complexity - inverse_temperature_beta * accuracy
    
    # Decomposition 2: F = Energy - Entropy  
    free_energy_2 = energy - entropy
    
    # Use the accuracy/complexity decomposition as primary
    free_energy = free_energy_1
    
    # Verify both decompositions are approximately equal (for debugging)
    decomposition_difference = abs(free_energy_1 - free_energy_2)
    if decomposition_difference > 1e-10:
        print(f"Warning: VFE decompositions differ by {decomposition_difference:.12f}")
        print(f"  Accuracy/Complexity: F = {complexity:.6f} - {inverse_temperature_beta:.3f}×{accuracy:.6f} = {free_energy_1:.6f}")
        print(f"  Energy/Entropy: F = {energy:.6f} - {entropy:.6f} = {free_energy_2:.6f}")
        print(f"  Expected: Both should be equal per Blahut-Arimoto formulation")
    
    return free_energy, complexity, accuracy, energy, entropy


def compute_information_bottleneck_curve(past_probabilities,
                                       future_conditional_probabilities,
                                       beta_values,
                                       em_solver_function):
    """
    Compute the information bottleneck trade-off curve.
    
    Sweeps over β values and computes the optimal trade-off between
    compression (complexity) and prediction accuracy.
    
    Args:
        past_probabilities: P(past) distribution
        future_conditional_probabilities: P(future|past) empirical
        beta_values: Array of inverse temperature values to sweep
        em_solver_function: EM algorithm function
        
    Returns:
        dict: Arrays of complexity, accuracy, free_energy for each β
    """
    complexity_values = []
    accuracy_values = []
    free_energy_values = []
    
    # Previous solution for warm start
    previous_posterior = None
    previous_emission = None
    
    for beta in beta_values:
        # Solve IB problem for this β
        posterior, emission, free_energy, complexity, accuracy = em_solver_function(
            past_probabilities, 
            future_conditional_probabilities, 
            beta,
            init_emission=previous_emission,
            init_posterior=previous_posterior
        )
        
        # Store results
        complexity_values.append(complexity)
        accuracy_values.append(accuracy)
        free_energy_values.append(free_energy)
        
        # Use as warm start for next β
        previous_posterior = posterior
        previous_emission = emission
    
    return {
        'beta_values': beta_values,
        'complexity': np.array(complexity_values),
        'accuracy': np.array(accuracy_values), 
        'free_energy': np.array(free_energy_values)
    }


def compute_hessian_eigenvalue_for_state(target_state_index,
                                       past_probabilities,
                                       posterior_distribution,
                                       emission_probabilities,
                                       future_conditional_probabilities):
    """
    Compute largest eigenvalue of Hessian matrix for state splitting analysis.
    
    Computes λ_max of the matrix:
    Σ_k = ∑_i P(past=i) q(state=k|past=i) (r_i - r_k)(r_i - r_k)ᵀ
    where r_i = P(future|past=i) and r_k = emission[k]
    
    This eigenvalue determines linear stability of the state - when β*λ_max > 1,
    the state becomes unstable and should be split.
    
    Args:
        target_state_index: Index of state to analyze
        past_probabilities: P(past) marginal [N_past]
        posterior_distribution: q(state|past) [N_past, N_states]
        emission_probabilities: p(future|state) [N_states, N_future]
        future_conditional_probabilities: P(future|past) [N_past, N_future]
        
    Returns:
        float: Largest eigenvalue λ_max
    """
    # Difference vectors: empirical vs emission for target state
    difference_vectors = (future_conditional_probabilities - 
                         emission_probabilities[target_state_index])  # [N_past, N_future]
    
    # Weights: P(past) * q(target_state|past)
    weights = (past_probabilities * 
              posterior_distribution[:, target_state_index])  # [N_past]
    
    # Weighted covariance matrix in the probability simplex
    covariance_matrix = (weights[:, None] * difference_vectors).T @ difference_vectors
    # Shape: [N_future, N_future]
    
    # Compute all eigenvalues and return the largest
    eigenvalues = np.linalg.eigvalsh(covariance_matrix)
    return eigenvalues.max()


def compute_phase_transition_indicators(beta_values, state_counts, free_energy_values):
    """
    Compute indicators of phase transitions in the IB solution.
    
    Phase transitions occur when the number of states changes discontinuously
    or when the free energy has non-smooth behavior.
    
    Args:
        beta_values: Array of inverse temperatures
        state_counts: Number of states at each β
        free_energy_values: Free energy at each β
        
    Returns:
        dict: Phase transition analysis results
    """
    # Convert to log scale for uniform spacing
    log_beta_values = np.log(beta_values)
    
    # Compute discrete derivatives
    state_transitions = np.diff(state_counts)
    free_energy_gradient = np.gradient(free_energy_values, log_beta_values)
    free_energy_curvature = np.gradient(free_energy_gradient, log_beta_values)
    
    # Find transition points (where state count changes)
    transition_indices = np.where(state_transitions != 0)[0]
    transition_beta_values = beta_values[transition_indices] if len(transition_indices) > 0 else []
    
    return {
        'transition_beta_values': transition_beta_values,
        'free_energy_gradient': free_energy_gradient,
        'free_energy_curvature': free_energy_curvature,
        'state_transition_points': transition_indices
    }


def compute_effective_dimension(posterior_distribution, epsilon=1e-8):
    """
    Compute effective dimensionality of the posterior distribution.
    
    Measures how many states are effectively being used by computing
    the exponential of the entropy: D_eff = exp(H[q(state|past)])
    
    Args:
        posterior_distribution: q(state|past) [N_past, N_states]
        epsilon: Small constant for numerical stability
        
    Returns:
        float: Effective dimension (1 to N_states)
    """
    # Clip for numerical stability
    posterior_clipped = np.clip(posterior_distribution, epsilon, 1.0)
    
    # Compute entropy of posterior for each past context
    entropy_per_context = -np.sum(posterior_clipped * np.log(posterior_clipped), axis=1)
    
    # Average entropy across contexts
    average_entropy = np.mean(entropy_per_context)
    
    # Effective dimension
    return np.exp(average_entropy)


if __name__ == "__main__":
    # Demonstration with synthetic data
    np.random.seed(42)
    
    # Create synthetic distributions
    num_past_contexts = 8
    num_states = 3
    num_future_symbols = 4
    
    past_probs = np.random.dirichlet([1] * num_past_contexts)
    posterior = np.random.dirichlet([1] * num_states, size=num_past_contexts)
    emission = np.random.dirichlet([1] * num_future_symbols, size=num_states)
    future_conditional = np.random.dirichlet([1] * num_future_symbols, size=num_past_contexts)
    
    # Test free energy computation
    beta = 2.0
    free_energy, complexity, accuracy, energy, entropy = compute_variational_free_energy(
        past_probs, posterior, emission, beta, future_conditional)
    
    print(f"Free energy: {free_energy:.3f}")
    print(f"Complexity: {complexity:.3f}")  
    print(f"Accuracy: {accuracy:.3f}")
    
    # Test eigenvalue computation
    max_eigenvalue = compute_hessian_eigenvalue_for_state(
        0, past_probs, posterior, emission, future_conditional)
    print(f"Max eigenvalue for state 0: {max_eigenvalue:.3f}")
    
    # Test effective dimension
    eff_dim = compute_effective_dimension(posterior)
    print(f"Effective dimension: {eff_dim:.3f}")