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


# Sequence-level reward implementation.
def ljp_reward_fn_seq_level(data_source, solution_str, ground_truth, extra_info=None):
    # ========== R_legal ==========
    ground_truth = json.loads(ground_truth)
    output = parse_output(solution_str)

    if output is None or not output:
        return 0.0

    if ('法条' in output and isinstance(output['法条'], dict)
            and '罪名' in output and isinstance(output['罪名'], list)
            and '刑期' in output and isinstance(output['刑期'], int)):
        format_score = 1.0
    else:
        format_score = 0.0

    if '法条' in output and isinstance(output['法条'], dict):
        articles = [(idx[:3] if len(idx) > 3 else idx) for idx in output['法条'].keys()]
        article_score = compute_f1_score(set(articles), set(ground_truth['法条']))
    else:
        article_score = 0.0

    if '罪名' in output and isinstance(output['罪名'], list):
        accusation_score = compute_f1_score(set(output['罪名']), set(ground_truth['罪名']))
    else:
        accusation_score = 0.0

    if '刑期' in output and isinstance(output['刑期'], int):
        term_score = relative_score(output['刑期'], ground_truth['刑期'])
    else:
        term_score = 0.0

    r_legal = format_score + 3 * article_score + 3 * accusation_score + 5 * term_score # [0, 12]

    # ========== R_structure ==========
    r_structure =  (int('[FACT]' in solution_str) +
                    int('[ARTICLE]' in solution_str) +
                    int('[CHARGE]' in solution_str) +
                    int('[SENTENCE]' in solution_str))
    r_structure /= 4 # [0, 1]

    # ========== R_consistency ==========
    if '法条' in output and isinstance(output['法条'], dict) and output['法条']:
        fact = solution_str.split('<reason>')[1].split('</reason>')[0]

        if '[FACT]' in fact:
            fact = fact.split('[FACT]')[1]
        if '[ARTICLE]' in fact:
            fact = fact.split('[ARTICLE]')[0]

        consistency_fact_article_ = consistency_fact_article(fact, output['法条'].keys()) # [0, 1]
    else:
        consistency_fact_article_ = 0.0

    if '法条' in output and isinstance(output['法条'], dict) and output['法条'] \
            and '罪名' in output and isinstance(output['罪名'], list) and output['罪名']:
        consistency_articles_accusations_ = consistency_articles_accusations(output['法条'].keys(), output['罪名']) # [0, 1]
    else:
        consistency_articles_accusations_ = 0.0

    if '罪名' in output and isinstance(output['罪名'], list) and output['罪名'] \
            and '刑期' in output and isinstance(output['刑期'], int):
        consistency_accusations_term_ = consistency_accusations_term(output['罪名'], output['刑期']) # [0, 1]
    else:
        consistency_accusations_term_ = 0.0

    r_consistency = consistency_fact_article_ + consistency_articles_accusations_ + consistency_accusations_term_ # [0, 3]

    return r_legal + r_structure + r_consistency


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

    # ========== R_legal ==========
    ground_truth = json.loads(ground_truth)
    output = parse_output(solution_str)

    if output is None or not output:
        return 0.0, [0.0 for _ in range(len(response_ids))]

    if ('法条' in output and isinstance(output['法条'], dict)
            and '罪名' in output and isinstance(output['罪名'], list)
            and '刑期' in output and isinstance(output['刑期'], int)):
        format_score = 1.0
    else:
        format_score = 0.0

    if '法条' in output and isinstance(output['法条'], dict):
        articles = [(idx[:3] if len(idx) > 3 else idx) for idx in output['法条'].keys()]
        article_score = compute_f1_score(set(articles), set(ground_truth['法条']))
    else:
        article_score = 0.0

    if '罪名' in output and isinstance(output['罪名'], list):
        accusation_score = compute_f1_score(set(output['罪名']), set(ground_truth['罪名']))
    else:
        accusation_score = 0.0

    if '刑期' in output and isinstance(output['刑期'], int):
        term_score = relative_score(output['刑期'], ground_truth['刑期'])
    else:
        term_score = 0.0

    r_legal = format_score + 3 * article_score + 3 * accusation_score + 5 * term_score  # [0, 12]

    # ========== R_structure ==========
    r_structure = (int('[FACT]' in solution_str) +
                   int('[ARTICLE]' in solution_str) +
                   int('[CHARGE]' in solution_str) +
                   int('[SENTENCE]' in solution_str))
    r_structure /= 4  # [0, 1]

    # ========== R_consistency ==========
    if '法条' in output and isinstance(output['法条'], dict) and output['法条']:
        fact = solution_str.split('<reason>')[1].split('</reason>')[0]

        if '[FACT]' in fact:
            fact = fact.split('[FACT]')[1]
        if '[ARTICLE]' in fact:
            fact = fact.split('[ARTICLE]')[0]

        consistency_fact_article_ = consistency_fact_article(fact, output['法条'].keys())  # [0, 1]
    else:
        consistency_fact_article_ = 0.0

    if '法条' in output and isinstance(output['法条'], dict) and output['法条'] \
            and '罪名' in output and isinstance(output['罪名'], list) and output['罪名']:
        consistency_articles_accusations_ = consistency_articles_accusations(output['法条'].keys(), output['罪名'])  # [0, 1]
    else:
        consistency_articles_accusations_ = 0.0

    if '罪名' in output and isinstance(output['罪名'], list) and output['罪名'] \
            and '刑期' in output and isinstance(output['刑期'], int):
        consistency_accusations_term_ = consistency_accusations_term(output['罪名'], output['刑期'])  # [0, 1]
    else:
        consistency_accusations_term_ = 0.0

    r_consistency = consistency_fact_article_ + consistency_articles_accusations_ + consistency_accusations_term_  # [0, 3]

    # ========== Compute the EntropyNorm for each token. ==========
    entropy_per_token = entropy_info['entropy_per_token'] # [response_len]

    max_entropy = max(entropy_per_token)
    entropy_norm_per_token = [entropy / max_entropy for entropy in entropy_per_token]  # 归一化

    # ========== Compute the LogicWeight for each token. ==========
    step_markers = [
        "[FACT]",
        "[ARTICLE]",
        "[CHARGE]",
        "[SENTENCE]"
    ]
    segments = parse_reasoning_segments(response_ids, tokenizer, step_markers)
    logic_weight_per_token = [r_consistency / 3 for _ in range(len(response_ids))]
    if '[ARTICLE]' in segments:
        start_pos, end_pos = segments['[ARTICLE]']
        logic_weight_per_token[start_pos: end_pos] = [consistency_fact_article_] * (end_pos - start_pos)
    if '[CHARGE]' in segments:
        start_pos, end_pos = segments['[CHARGE]']
        logic_weight_per_token[start_pos: end_pos] = [consistency_articles_accusations_] * (end_pos - start_pos)
    if '[SENTENCE]' in segments:
        start_pos, end_pos = segments['[SENTENCE]']
        logic_weight_per_token[start_pos: end_pos] = [consistency_accusations_term_] * (end_pos - start_pos)

    # ========== Compute the weight for each token. ==========
    weight_per_token = [alpha * entropy_norm + (1 - alpha) * logic_weight
                        for entropy_norm, logic_weight in zip(entropy_norm_per_token, logic_weight_per_token)]

    weight_per_token_mean = statistics.mean(weight_per_token)
    weight_per_token = [weight - weight_per_token_mean for weight in weight_per_token] # make the mean zero

    return r_legal + r_structure + r_consistency, weight_per_token