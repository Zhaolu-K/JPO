import json
import math
from typing import List
import jieba
import numpy as np
from scipy import stats
from rl.legal_naive_bayes import LegalNaiveBayes

# Load the JSON file of the criminal law.
with open(f'../data/law.json', 'r', encoding='utf-8') as file:
    law = json.load(file)

# Load the mapping file from charges to the mean and standard deviation of prison sentences.
with open(f'accusation_to_term_mean_std.json', 'r', encoding='utf-8') as file:
    accusation_to_term_mean_std = json.load(file)

# Load the Naive Bayes model constructed from the mapping relationships between case facts and legal provisions in the training dataset.
legal_naive_bayes = LegalNaiveBayes()
legal_naive_bayes.load_model(f'legal_naive_bayes.joblib')


# Parse string to JSON object.
def json_parse(json_str: str):
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None


# Parse legal judgement result from model's output.
def parse_output(output: str):
    if not ('<reason>' in output and '</reason>' in output and '<answer>' in output and '</answer>' in output):
        return None
    output = output.split('<answer>')[1].split('</answer>')[0].strip()
    return json_parse(output)


# Calculate the F1 score between the predicted set and the ground truth set.
def compute_f1_score(predict: set, ground_truth: set) -> float:
    intersection = predict & ground_truth
    precision = len(intersection) / len(predict) if predict else 0.0
    recall = len(intersection) / len(ground_truth) if ground_truth else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0
    return f1_score


# Compute the exponentially decaying form of the relative error between the predicted value and the true value.
def relative_score(predicted: int, true: int, alpha: float = 3.0) -> float:
    if true == 0:
        return 0.0 if predicted != 0 else 1.0
    relative_error = abs(predicted - true) / true
    return math.exp(-alpha * relative_error)


# Compute the logical consistency from facts to legal articles.
def consistency_fact_article(fact: str, article_idxs: List[str]) -> float:
    article_idxs = [idx[:3] if len(idx) > 3 else idx for idx in article_idxs]
    return legal_naive_bayes.compute_P_A_given_F(fact, article_idxs)


# Compute the logical consistency from legal articles to charges.
def consistency_articles_accusations(article_idxs: List[str], accusations: List[str]) -> float:
    article_idxs = [idx[:3] if len(idx) > 3 else idx for idx in article_idxs]

    def consistency_article_accusation(idx: str, accusation: str) -> float:
        article_contents = []
        if idx in law:
            article_contents.append(law[idx])
            i = 1
            while f'{idx}.{i}' in law:
                article_contents.append(law[f'{idx}.{i}'])
                i += 1

        if not article_contents:
            return 0.0

        article_contents = ''.join(article_contents)
        if accusation in article_contents:
            return 2.0

        accusation_word_list = jieba.lcut(accusation)
        accusation_word_list = [word for word in accusation_word_list if len(word) >= 2]

        if not accusation_word_list:
            return 0.0
        
        cnt = 0
        for word in accusation_word_list:
            if word in article_contents:
                cnt += 1
        
        if cnt / len(accusation_word_list) >= 0.3:
            return 1.0
        
        return 0.0

    consistency_matrix = np.array(
        [[consistency_article_accusation(idx, accusation) for accusation in accusations] for idx in article_idxs]
    )

    return np.sum(consistency_matrix.max(axis=1)) / (2.0 * len(article_idxs))


# Compute the logical consistency from charges to sentence.
def consistency_accusations_term(accusations: List[str], term: int):
    tot_consistency_value = 0.0
    cnt = 0
    for accusation in accusations:
        if accusation in accusation_to_term_mean_std:
            mean = accusation_to_term_mean_std[accusation]['mean']
            std = accusation_to_term_mean_std[accusation]['std']
            norm_dist = stats.norm(loc=mean, scale=std)
            pdf_value_term = norm_dist.pdf(term)
            pdf_value_max = norm_dist.pdf(mean)
            consistency_value = pdf_value_term / pdf_value_max
            tot_consistency_value += consistency_value
            cnt += 1
    return tot_consistency_value / cnt if cnt > 0 else 0.0