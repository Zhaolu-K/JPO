import json
import pandas as pd
import prompts

TRAIN_DATA_PATH = '../data/jpo_dataset/train_rl.json'
TEST_DATA_PATH = '../data/jpo_dataset/test.json'
DATA_SAVE_PATH = '../data/preprocessed_data'

MAX_TOKEN_FACT = 800

def build_rl_dataset(type: str = "TRAIN"):
    if type == "TRAIN":
        data_path = TRAIN_DATA_PATH
    else:
        data_path = TEST_DATA_PATH
    
    dataset = []

    with open(data_path, 'r', encoding='utf-8') as f:
        idx = 0
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            if len(data['fact']) <= MAX_TOKEN_FACT:
                data_in_verl_format = {
                    "data_source": 'jpo-dataset',
                    "prompt": [{
                        "role": "user",
                        "content": f"{prompts.INSTRUCTION_STUDENT_MODEL}\n\n{data["fact"]}"
                    }],
                    "ability": "Legal Judgment Prediction",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": json.dumps({
                            '法条': data['meta']['relevant_articles'],
                            '罪名': data['meta']['accusation'],
                            '刑期': data['meta']['term_of_imprisonment']['imprisonment']
                        }, ensure_ascii=False)
                    },
                    "extra_info": {
                        "split": type,
                        "index": idx
                    }
                }
                dataset.append(data_in_verl_format)
                idx += 1

    print("=" * 10 + f" {type} DATASET INFO " + "=" * 10)
    print(f"dataset size: {len(dataset)}")

    if dataset:
        print(f"example:\n\n{dataset[0]}\n\n")

    return dataset

def main():
    train_dataset = build_rl_dataset("TRAIN")
    test_dataset = build_rl_dataset("TEST")

    with open(f'{DATA_SAVE_PATH}/train_rl.json', 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, ensure_ascii=False, indent=2)

    with open(f'{DATA_SAVE_PATH}/test_rl.json', 'w', encoding='utf-8') as f:
        json.dump(test_dataset, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(train_dataset)
    df.to_parquet(f'{DATA_SAVE_PATH}/train_rl.parquet', index=False)

    df = pd.DataFrame(test_dataset)
    df.to_parquet(f'{DATA_SAVE_PATH}/test_rl.parquet', index=False)

if __name__ == '__main__':
    main()