import json
import pandas as pd
import prompts

MAX_TOKEN_FACT = 800

with open(f'../data/preprocessed_data/train_sft.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# change format
final_data = []

for item in data:
    item_2 = {
        'question': f'{prompts.INSTRUCTION}\n\n{item["fact"]}',
        'answer': f'<reason>\n{item["chain_of_thought"]}\n</reason>\n<answer>\n{item["answer"]}\n</answer>'
    }
    final_data.append(item_2)

df = pd.DataFrame(final_data)
df.to_parquet(f'../data/preprocessed_data/train_sft.parquet', index=False)


test_data = []
with open('../data/jpo_dataset/test.json', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        data = json.loads(line.strip())
        if len(data['fact']) <= MAX_TOKEN_FACT:
            data_2 = dict()
            data_2['question'] = f'{prompts.INSTRUCTION}\n\n{data["fact"]}'
            data_2['answer'] = json.dumps({
                '法条': data['meta']['relevant_articles'],
                '罪名': data['meta']['accusation'],
                '刑期': data['meta']['term_of_imprisonment']['imprisonment']
            }, ensure_ascii=False)
            test_data.append(data_2)
print(len(test_data))
df = pd.DataFrame(test_data)
df.to_parquet(f'../data/preprocessed_data/test_sft.parquet', index=False)