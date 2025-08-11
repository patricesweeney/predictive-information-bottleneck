"""
Empirical Probability Analysis for Sequential Data
=================================================

Tools for extracting empirical conditional probabilities from sequential data.
Used to estimate P(future | past) from observed sequences for information
bottleneck analysis.
"""

import numpy as np
from collections import defaultdict


def extract_empirical_future_given_past(sequence, past_window_length=10, future_window_length=2):
    """
    Extract empirical conditional probabilities P(future | past) from sequence data.
    
    Divides sequence into overlapping windows and counts transitions to estimate
    the conditional distribution of future symbols given past context.
    
    Args:
        sequence: Binary sequence array
        past_window_length: Length of past context window (L_past)
        future_window_length: Length of future prediction window (L_future)
        
    Returns:
        tuple: (past_words, past_probabilities, future_conditional_probabilities)
            - past_words: List of observed past context indices
            - past_probabilities: P(past) for each observed context
            - future_conditional_probabilities: P(future|past) matrix
    """
    alphabet_size = 2 ** future_window_length
    
    # Count co-occurrences of (past, future) pairs
    frequency_table = defaultdict(lambda: np.zeros(alphabet_size, int))
    past_occurrence_counts = defaultdict(int)
    
    # Scan through sequence extracting overlapping windows
    sequence_length = len(sequence)
    for time_index in range(past_window_length, sequence_length - future_window_length + 1):
        # Extract past context window
        past_window = sequence[time_index - past_window_length:time_index]
        past_index = convert_binary_window_to_index(past_window)
        
        # Extract future prediction window  
        future_window = sequence[time_index:time_index + future_window_length]
        future_index = convert_binary_window_to_index(future_window)
        
        # Update counts
        frequency_table[past_index][future_index] += 1
        past_occurrence_counts[past_index] += 1
    
    # Convert to sorted lists and probability arrays
    observed_past_words = sorted(frequency_table.keys())
    
    # Marginal probability P(past)
    past_probabilities = np.array([past_occurrence_counts[past_word] 
                                  for past_word in observed_past_words], dtype=float)
    past_probabilities /= past_probabilities.sum()
    
    # Conditional probabilities P(future | past)
    future_conditional_probabilities = np.array([
        frequency_table[past_word] / frequency_table[past_word].sum() 
        for past_word in observed_past_words
    ])
    
    return observed_past_words, past_probabilities, future_conditional_probabilities


def convert_binary_window_to_index(binary_window):
    """
    Convert binary sequence window to integer index.
    
    Args:
        binary_window: Array of 0s and 1s
        
    Returns:
        int: Binary representation interpreted as integer
    """
    binary_string = ''.join(map(str, binary_window))
    return int(binary_string, 2)


def convert_index_to_binary_window(index, window_length):
    """
    Convert integer index back to binary window.
    
    Args:
        index: Integer representation
        window_length: Desired length of binary window
        
    Returns:
        np.array: Binary sequence of specified length
    """
    binary_string = format(index, f'0{window_length}b')
    return np.array([int(bit) for bit in binary_string])


def compute_empirical_entropy(probabilities, base=2):
    """
    Compute empirical entropy H(X) = -∑ p(x) log p(x).
    
    Args:
        probabilities: Probability distribution array
        base: Logarithm base (2 for bits, e for nats)
        
    Returns:
        float: Entropy value
    """
    # Handle zero probabilities gracefully
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    
    if base == 2:
        return -np.sum(probabilities * np.log2(probabilities))
    else:
        return -np.sum(probabilities * np.log(probabilities))


def compute_empirical_conditional_entropy(joint_probabilities, marginal_probabilities, base=2):
    """
    Compute empirical conditional entropy H(Y|X) = -∑∑ p(x,y) log p(y|x).
    
    Args:
        joint_probabilities: P(X,Y) joint distribution matrix
        marginal_probabilities: P(X) marginal distribution
        base: Logarithm base
        
    Returns:
        float: Conditional entropy value
    """
    # Compute conditional probabilities P(Y|X)
    conditional_probabilities = joint_probabilities / (marginal_probabilities[:, None] + 1e-12)
    conditional_probabilities = np.clip(conditional_probabilities, 1e-12, 1.0)
    
    if base == 2:
        log_conditional = np.log2(conditional_probabilities)
    else:
        log_conditional = np.log(conditional_probabilities)
    
    return -np.sum(joint_probabilities * log_conditional)


def compute_empirical_mutual_information(past_probabilities, future_conditional_probabilities, base=2):
    """
    Compute empirical mutual information I(Past; Future).
    
    Args:
        past_probabilities: P(past) marginal distribution
        future_conditional_probabilities: P(future|past) conditional distribution
        base: Logarithm base
        
    Returns:
        float: Mutual information value
    """
    # Joint distribution P(past, future)
    joint_probabilities = past_probabilities[:, None] * future_conditional_probabilities
    
    # Marginal distribution P(future) 
    future_marginal_probabilities = joint_probabilities.sum(axis=0)
    
    # Compute MI = H(Future) - H(Future|Past)
    future_entropy = compute_empirical_entropy(future_marginal_probabilities, base)
    conditional_entropy = compute_empirical_conditional_entropy(
        joint_probabilities, past_probabilities, base)
    
    return future_entropy - conditional_entropy


def estimate_statistical_complexity(sequence, max_window_length=15):
    """
    Estimate statistical complexity by measuring information in past contexts.
    
    Statistical complexity Cμ quantifies the amount of information about the past
    needed to optimally predict the future.
    
    Args:
        sequence: Binary sequence for analysis
        max_window_length: Maximum past window length to consider
        
    Returns:
        dict: Window lengths and corresponding complexity estimates
    """
    complexity_estimates = {}
    
    for window_length in range(1, max_window_length + 1):
        # Extract past contexts
        past_words, past_probabilities, _ = extract_empirical_future_given_past(
            sequence, past_window_length=window_length, future_window_length=1)
        
        # Complexity ≈ entropy of past contexts
        complexity = compute_empirical_entropy(past_probabilities)
        complexity_estimates[window_length] = complexity
    
    return complexity_estimates


if __name__ == "__main__":
    # Demonstration with synthetic data
    from processes import PROCESS_GENERATORS
    
    # Generate test sequence
    generator = PROCESS_GENERATORS["Golden-Mean"]
    test_sequence = generator(1000, seed=42)
    
    # Extract empirical probabilities
    past_words, past_probs, future_conditional = extract_empirical_future_given_past(
        test_sequence, past_window_length=3, future_window_length=2)
    
    print(f"Number of observed past contexts: {len(past_words)}")
    print(f"Past context probabilities: {past_probs[:5]}...")  # Show first 5
    print(f"Future conditional shape: {future_conditional.shape}")
    
    # Compute information measures
    mutual_info = compute_empirical_mutual_information(past_probs, future_conditional)
    print(f"Mutual information I(Past; Future): {mutual_info:.3f} bits")
    
    # Estimate statistical complexity
    complexity_estimates = estimate_statistical_complexity(test_sequence)
    print(f"Complexity estimates by window length: {complexity_estimates}")