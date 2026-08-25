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

# Load model's response on test dataset.
with open(predict_path, 'r', encoding='utf-8') as f:
    predict = json.load(f)

len_data = len(predict) # size of test dataset
len_valid_predict = len_data # size of valid response (can be parsed by JSON and not null) on test dataset
len_correct_format = len_data # size of response in correct format on test dataset


def main():
    acc_article = 0.0
    acc_accusation = 0.0
    acc_term = 0.0
    correct_term = 0
    incorrect_term = 0
    r_consistency = 0.0

    for idx, item in enumerate(predict):
        output = parse_output(item['model_output'])
        answer = item['answer']

        if output is None or not output:
            len_valid_predict -= 1
            len_correct_format -= 1
            continue

        # accuracy of final result
        if '法条' in output and isinstance(output['法条'], dict):
            articles = [(idx[:3] if len(idx) > 3 else idx) for idx in output['法条'].keys()]
            acc_article += compute_f1_score(set(articles), set(answer['法条']))
        if '罪名' in output and isinstance(output['罪名'], list):
            acc_accusation += compute_f1_score(set(output['罪名']), set(answer['罪名']))
        if '刑期' in output and isinstance(output['刑期'], int):
            acc_term += relative_score(output['刑期'], answer['刑期'])
            if output['刑期'] == answer['刑期']:
                correct_term += 1
            else:
                incorrect_term += 1

        # consistency of fact to article
        if '法条' in output and isinstance(output['法条'], dict) and output['法条']:
            fact = item['model_output'].split('<reason>')[1].split('</reason>')[0]
        
            if '[FACT]' in fact:
                fact = fact.split('[FACT]')[1]
            if '[ARTICLE]' in fact:
                fact = fact.split('[ARTICLE]')[0]
        
            consistency_fact_article_ = consistency_fact_article(fact, output['法条'].keys())  # [0, 1]

        else:
            consistency_fact_article_ = 0.0
        
        # consistency of article to charge
        if '法条' in output and isinstance(output['法条'], dict) and output['法条'] \
                and '罪名' in output and isinstance(output['罪名'], list) and output['罪名']:
            consistency_articles_accusations_ = consistency_articles_accusations(output['法条'].keys(),
                                                                                output['罪名'])  # [0, 1]
        else:
            consistency_articles_accusations_ = 0.0
        
        # consistency of charge to sentence
        if '罪名' in output and isinstance(output['罪名'], list) and output['罪名'] \
                and '刑期' in output and isinstance(output['刑期'], int):
            consistency_accusations_term_ = consistency_accusations_term(output['罪名'], output['刑期'])  # [0, 1]
        else:
            consistency_accusations_term_ = 0.0
        
        # consistency score
        r_consistency += (consistency_fact_article_ + consistency_articles_accusations_ + consistency_accusations_term_) / 3

        if not ('法条' in output and isinstance(output['法条'], dict)
        and '罪名' in output and isinstance(output['罪名'], list)
        and '刑期' in output and isinstance(output['刑期'], int)):
            len_correct_format -=1

    print(f"""Size of test dataset: {len_data}
    The number of outputs with the correct format: {len_correct_format}
    The proportion of outputs with the correct format: {len_correct_format / len_data}
    Average prediction score of articles (F1 Score): {acc_article / len_valid_predict}
    Average prediction score of charges (F1 Score): {acc_accusation / len_valid_predict}
    Average prediction score of charges (relative error): {acc_term / len_valid_predict}
    The number of correct predictions of sentencing: {correct_term}
    The number of incorrect predictions of sentencing: {incorrect_term}
    The proportion of correct predictions of sentencing: {correct_term / (correct_term + incorrect_term)}
    Average consistency score: {r_consistency / len_data}""")


if __name__ == '__main__':
    main()