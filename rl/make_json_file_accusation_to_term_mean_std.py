import json
import statistics

SOURCE_PATH_SFT = '../data/jpo_dataset/train_sft.json'
SOURCE_PATH_RL = '../data/jpo_dataset/train_rl.json'
SAVE_PATH = 'accusation_to_term_mean_std.json'

source_data = []

with open(SOURCE_PATH_SFT, 'r', encoding='utf-8') as file:
    for line in file:
        if line.strip():
            data = json.loads(line)
            source_data.append(data)

with open(SOURCE_PATH_RL, 'r', encoding='utf-8') as file:
    for line in file:
        if line.strip():
            data = json.loads(line)
            source_data.append(data)

accusation_to_terms = dict() # Dict[str, List[int]]

for item in source_data:
    accusations = item['meta']['accusation']
    term = item['meta']['term_of_imprisonment']['imprisonment']
    for accusation in accusations:
        if accusation not in accusation_to_terms:
            accusation_to_terms[accusation] = [term]
        else:
            accusation_to_terms[accusation].append(term)

accusation_to_term_mean_std = dict()
for accusation, terms in accusation_to_terms.items():
    mean = statistics.mean(terms)
    std = statistics.pstdev(terms)
    accusation_to_term_mean_std[accusation] = {'mean': mean, 'std': std}

with open(SAVE_PATH, 'w', encoding='utf-8') as file:
    json.dump(accusation_to_term_mean_std, file, ensure_ascii=False, indent=4)