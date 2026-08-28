import json

from metrics import (
    compute_f1_score, 
    relative_score, 
    consistency_fact_article, 
    consistency_articles_accusations, 
    consistency_accusations_term,
    parse_output
)

predict_path = 'data/qwen2_5_3b/qwen2_5_3b_predict.json'

ARTICLE_KEY = '法条'
ACCUSATION_KEY = '罪名'
TERM_KEY = '刑期'


def _articles(output):
    """截断到 3 位的法条编号集合；字段缺失或类型不符时返回 None，空 dict 返回空集合。"""
    value = output.get(ARTICLE_KEY)
    if not isinstance(value, dict):
        return None
    return {(idx[:3] if len(idx) > 3 else idx) for idx in value.keys()}


def _accusations(output):
    value = output.get(ACCUSATION_KEY)
    return set(value) if isinstance(value, list) else None


def _term(output):
    """刑期，可能是 0，因此调用方须用 `is not None` 判断。"""
    value = output.get(TERM_KEY)
    return value if isinstance(value, int) else None


def _has_correct_format(output):
    return (_articles(output) is not None
            and _accusations(output) is not None
            and _term(output) is not None)


def _extract_fact(model_output):
    """从 <reason> 段落里截出事实描述部分。"""
    fact = model_output.split('<reason>')[1].split('</reason>')[0]
    if '[FACT]' in fact:
        fact = fact.split('[FACT]')[1]
    if '[ARTICLE]' in fact:
        fact = fact.split('[ARTICLE]')[0]
    return fact


def _prediction_scores(output, answer):
    """(法条 F1, 罪名 F1, 刑期相对分)，缺失的项记 0。"""
    articles = _articles(output)
    accusations = _accusations(output)
    term = _term(output)

    article_score = 0.0 if articles is None else compute_f1_score(articles, set(answer[ARTICLE_KEY]))
    accusation_score = 0.0 if accusations is None else compute_f1_score(accusations, set(answer[ACCUSATION_KEY]))
    term_score = 0.0 if term is None else relative_score(term, answer[TERM_KEY])
    return article_score, accusation_score, term_score


def _consistency_scores(output, model_output):
    """(事实→法条, 法条→罪名, 罪名→刑期) 三段一致性，各自 [0, 1]。

    一致性要求字段非空，且用的是未截断的原始法条编号。
    """
    articles = _articles(output)
    accusations = _accusations(output)
    term = _term(output)
    raw_article_keys = output[ARTICLE_KEY].keys() if articles else None

    if articles:
        fact_article = consistency_fact_article(_extract_fact(model_output), raw_article_keys)
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


class _Totals:
    """遍历测试集时的累加器。"""

    def __init__(self, size):
        self.size = size                # 测试集规模
        self.valid = size               # 能被 JSON 解析且非空的响应数
        self.correct_format = size      # 格式完全正确的响应数
        self.article = 0.0
        self.accusation = 0.0
        self.term = 0.0
        self.correct_term = 0
        self.incorrect_term = 0
        self.consistency = 0.0


def _accumulate(totals, item):
    output = parse_output(item['model_output'])
    answer = item['answer']

    if output is None or not output:
        totals.valid -= 1
        totals.correct_format -= 1
        return

    article_score, accusation_score, term_score = _prediction_scores(output, answer)
    totals.article += article_score
    totals.accusation += accusation_score
    totals.term += term_score

    term = _term(output)
    if term is not None:
        if term == answer[TERM_KEY]:
            totals.correct_term += 1
        else:
            totals.incorrect_term += 1

    totals.consistency += sum(_consistency_scores(output, item['model_output'])) / 3

    if not _has_correct_format(output):
        totals.correct_format -= 1


def _format_report(totals):
    return f"""Size of test dataset: {totals.size}
    The number of outputs with the correct format: {totals.correct_format}
    The proportion of outputs with the correct format: {totals.correct_format / totals.size}
    Average prediction score of articles (F1 Score): {totals.article / totals.valid}
    Average prediction score of charges (F1 Score): {totals.accusation / totals.valid}
    Average prediction score of charges (relative error): {totals.term / totals.valid}
    The number of correct predictions of sentencing: {totals.correct_term}
    The number of incorrect predictions of sentencing: {totals.incorrect_term}
    The proportion of correct predictions of sentencing: {totals.correct_term / (totals.correct_term + totals.incorrect_term)}
    Average consistency score: {totals.consistency / totals.size}"""


def main():
    with open(predict_path, 'r', encoding='utf-8') as f:
        predict = json.load(f)

    totals = _Totals(len(predict))
    for item in predict:
        _accumulate(totals, item)

    print(_format_report(totals))


if __name__ == '__main__':
    main()
