import numpy as np
from typing import List, Tuple
from itertools import combinations


def one_hot_encode_trajectories(trajectories: List[List[int]], num_actions: int) -> List[np.ndarray]:
    """
    One-hot encode a list of action trajectories.

    Args:
        trajectories: List of action trajectories (list of integers).
        num_actions: Total number of discrete actions.

    Returns:
        List of one-hot encoded action trajectories (each as 1D vector).
    """
    encoded = []
    for traj in trajectories:
        encoded_traj = []
        for a in traj:
            one_hot = np.zeros((num_actions,))
            one_hot[a] = 1
            encoded_traj.append(one_hot)
        encoded.append(np.concatenate(encoded_traj))  # flatten
    return encoded


def distance_lcs(seq1: List[int], seq2: List[int]) -> float:
    """
    Compute LCS-based distance between two sequences.

    Returns:
        A distance value in [0, 1].
    """
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n):
        for j in range(m):
            if seq1[i] == seq2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    lcs_length = dp[n][m]
    max_length = max(n, m)
    return 1.0 - lcs_length / max_length


def is_fuzzy_subsequence(sub: Tuple[int], seq: List[int]) -> bool:
    """
    Check if sub is a fuzzy (non-contiguous) subsequence of seq.
    """
    it = iter(seq)
    return all(item in it for item in sub)


def generate_candidates(seq: List[int], min_len: int):
    """
    Generate all fuzzy subsequence candidates of minimum length.
    """
    n = len(seq)
    for l in range(min_len, n + 1):
        for indices in combinations(range(n), l):
            yield tuple(seq[i] for i in indices)


def distance_fuzzy_lcs(seq1: List[int], seq2: List[int], min_len: int = 3) -> float:
    """
    Compute fuzzy LCS-based distance between two sequences.

    Returns:
        A distance value in [0, 1].
    """
    candidates = set(generate_candidates(seq1, min_len))
    best = ()
    best_count = -1
    best_len = -1
    best_diversity = -1

    for cand in candidates:
        count = sum(is_fuzzy_subsequence(cand, seq)
                    for seq in [seq1, seq2])
        diversity = len(set(cand))
        if (
            count > best_count
            or (count == best_count and len(cand) > best_len)
            or (count == best_count and len(cand) == best_len and diversity > best_diversity)
        ):
            best = cand
            best_count = count
            best_len = len(cand)
            best_diversity = diversity

    score = len(best) / max(len(seq1), len(seq2))
    return 1.0 - score


def smith_waterman(seq1: List[int], seq2: List[int],
                   match_score=2, mismatch_penalty=-1, gap_penalty=-1) -> Tuple[int, int, int]:
    """
    Compute Smith-Waterman score and alignment length.
    """
    m, n = len(seq1), len(seq2)
    H = np.zeros((m + 1, n + 1), dtype=int)
    max_score = 0
    max_pos = None

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = H[i - 1][j - 1] + \
                (match_score if seq1[i - 1] ==
                 seq2[j - 1] else mismatch_penalty)
            delete = H[i - 1][j] + gap_penalty
            insert = H[i][j - 1] + gap_penalty
            H[i][j] = max(0, match, delete, insert)

            if H[i][j] > max_score:
                max_score = H[i][j]
                max_pos = (i, j)

    # backtrack to find length of alignment
    i, j = max_pos
    align_length = 0
    while H[i][j] > 0:
        score = H[i][j]
        diag = H[i - 1][j - 1]
        up = H[i - 1][j]
        left = H[i][j - 1]

        if score == diag + (match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty):
            i -= 1
            j -= 1
        elif score == up + gap_penalty:
            i -= 1
        elif score == left + gap_penalty:
            j -= 1
        else:
            break
        align_length += 1

    normalized_score = max_score / \
        (match_score * max(min(len(seq1), len(seq2)), 1))
    return 1.0 - normalized_score  # Convert to distance


def distance_smith_waterman(seq1: List[int], seq2: List[int]) -> float:
    """
    Smith-Waterman based distance.
    """
    return smith_waterman(seq1, seq2)
