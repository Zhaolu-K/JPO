import json
from typing import List
import statistics
from metrics import (
    compute_f1_score, 
    relative_score, 
    consistency_fact_article, 
    consistency_articles_accusations, 
    consistency_accusations_term,
    parse_output
)

ARTICLE_KEY = '法条'
ACCUSATION_KEY = '罪名'
TERM_KEY = '刑期'

STEP_MARKERS = [
    "[FACT]",
    "[ARTICLE]",
    "[CHARGE]",
    "[SENTENCE]"
]


def parse_reasoning_segments(response_ids, tokenizer, step_markers):
    """
    Specifically parse each step of the chain of thought and return the token range for each step.

    step_markers = [
        "[FACT]",
        "[ARTICLE]",
        "[CHARGE]",
        "[SENTENCE]"
    ]
    """

    # Encode all step markers.
    step_ids = [tokenizer.encode(marker, add_special_tokens=False) for marker in step_markers]

    segments = {}
    response_len = len(response_ids)

    # Find the start position of each step.
    step_starts = []
    for i, marker_ids in enumerate(step_ids):
        marker_len = len(marker_ids)
        for j in range(response_len - marker_len + 1):
            if response_ids[j:j + marker_len] == marker_ids:
                step_starts.append((j, i))  # (position, step index)
                break

    # Sort by position.
    step_starts.sort()

    # Determine the range of each step.
    for idx, (start_pos, step_idx) in enumerate(step_starts):
        end_pos = step_starts[idx + 1][0] if idx + 1 < len(step_starts) else response_len
        segments[step_markers[step_idx]] = (start_pos, end_pos)  # Left-closed, right-open [start_pos, end_pos)

    return segments # {'[FACT]': (start_pos, end_pos), ...}


def _articles(output):
    """Article ids truncated to 3 chars; None when the field is absent or mistyped."""
    value = output.get(ARTICLE_KEY)
    if not isinstance(value, dict):
        return None
    return {(idx[:3] if len(idx) > 3 else idx) for idx in value.keys()}


def _accusations(output):
    value = output.get(ACCUSATION_KEY)
    return set(value) if isinstance(value, list) else None


def _term(output):
    """Prison term; may legitimately be 0, so callers must test `is not None`."""
    value = output.get(TERM_KEY)
    return value if isinstance(value, int) else None


def _extract_fact(solution_str):
    """Pull the fact description out of the <reason> block."""
    fact = solution_str.split('<reason>')[1].split('</reason>')[0]

    if '[FACT]' in fact:
        fact = fact.split('[FACT]')[1]
    if '[ARTICLE]' in fact:
        fact = fact.split('[ARTICLE]')[0]

    return fact


def _r_legal(output, ground_truth):
    """Legal prediction quality, [0, 12]."""
    articles = _articles(output)
    accusations = _accusations(output)
    term = _term(output)

    format_score = 1.0 if (articles is not None and accusations is not None and term is not None) else 0.0
    article_score = compute_f1_score(articles, set(ground_truth[ARTICLE_KEY])) if articles is not None else 0.0
    accusation_score = compute_f1_score(accusations, set(ground_truth[ACCUSATION_KEY])) if accusations is not None else 0.0
    term_score = relative_score(term, ground_truth[TERM_KEY]) if term is not None else 0.0

    return format_score + 3 * article_score + 3 * accusation_score + 5 * term_score


def _r_structure(solution_str):
    """Reasoning structure completeness, [0, 1]."""
    return sum(int(marker in solution_str) for marker in STEP_MARKERS) / len(STEP_MARKERS)


def _consistency_scores(output, solution_str):
    """(fact->article, article->charge, charge->sentence), each [0, 1].

    Consistency requires the fields to be non-empty and uses the untruncated
    article ids, unlike the prediction scores in `_r_legal`.
    """
    articles = _articles(output)
    accusations = _accusations(output)
    term = _term(output)
    raw_article_keys = output[ARTICLE_KEY].keys() if articles else None

    if articles:
        fact_article = consistency_fact_article(_extract_fact(solution_str), raw_article_keys)
    else:
        fact_article = 0.0

    if articles and accusations:
        article_accusation = consistency_articles_accusations(raw_article_keys, output[ACCUSATION_KEY])
    else:
        article_accusation = 0.0

    if accusations and term is not None:
        accusation_term = consistency_accusations_term(output[ACCUSATION_KEY], term)
    else:
        accusation_term = 0.0

    return fact_article, article_accusation, accusation_term


def _reward_components(solution_str, ground_truth):
    """Shared by both reward implementations.

    Returns (r_legal, r_structure, consistency_triple) or None when the model
    output cannot be parsed.
    """
    output = parse_output(solution_str)
    if output is None or not output:
        return None

    r_legal = _r_legal(output, ground_truth)
    r_structure = _r_structure(solution_str)
    consistency = _consistency_scores(output, solution_str)
    return r_legal, r_structure, consistency


# Sequence-level reward implementation.
def ljp_reward_fn_seq_level(data_source, solution_str, ground_truth, extra_info=None):
    ground_truth = json.loads(ground_truth)

    components = _reward_components(solution_str, ground_truth)
    if components is None:
        return 0.0

    r_legal, r_structure, consistency = components
    return r_legal + r_structure + sum(consistency)  # r_consistency in [0, 3]


def _entropy_norm_per_token(entropy_info):
    entropy_per_token = entropy_info['entropy_per_token']  # [response_len]
    max_entropy = max(entropy_per_token)
    return [entropy / max_entropy for entropy in entropy_per_token]  # 归一化


def _logic_weight_per_token(response_ids, tokenizer, consistency):
    """Per-token logic weight: the matching consistency score inside each
    reasoning step, and the mean consistency elsewhere."""
    fact_article, article_accusation, accusation_term = consistency
    segments = parse_reasoning_segments(response_ids, tokenizer, STEP_MARKERS)

    weights = [sum(consistency) / 3 for _ in range(len(response_ids))]
    for marker, score in (("[ARTICLE]", fact_article),
                          ("[CHARGE]", article_accusation),
                          ("[SENTENCE]", accusation_term)):
        if marker in segments:
            start_pos, end_pos = segments[marker]
            weights[start_pos: end_pos] = [score] * (end_pos - start_pos)

    return weights


# Token-level reward implementation, return sequence-level reward and importance weight per token.
def ljp_reward_fn_token_level(
    data_source,
    solution_str: str,
    ground_truth: str,
    response_ids: List,
    tokenizer,
    vocab_size,
    entropy_info,
    alpha: float = 0.6,
    extra_info = None
):
    """
        Args:
            entropy_info = {
                "entropy_mean": seq_entropy.mean().item(),
                "entropy_max": seq_entropy.max().item(),
                "entropy_min": seq_entropy.min().item(),
                "entropy_std": seq_entropy.std().item(),
                "entropy_per_token": seq_entropy.cpu().numpy().tolist(),
            }
    """
    ground_truth = json.loads(ground_truth)

    components = _reward_components(solution_str, ground_truth)
    if components is None:
        return 0.0, [0.0 for _ in range(len(response_ids))]

    r_legal, r_structure, consistency = components

    entropy_norm_per_token = _entropy_norm_per_token(entropy_info)
    logic_weight_per_token = _logic_weight_per_token(response_ids, tokenizer, consistency)

    # ========== Compute the weight for each token. ==========
    weight_per_token = [alpha * entropy_norm + (1 - alpha) * logic_weight
                        for entropy_norm, logic_weight in zip(entropy_norm_per_token, logic_weight_per_token)]

    weight_per_token_mean = statistics.mean(weight_per_token)
    weight_per_token = [weight - weight_per_token_mean for weight in weight_per_token] # make the mean zero

    return r_legal + r_structure + sum(consistency), weight_per_token
