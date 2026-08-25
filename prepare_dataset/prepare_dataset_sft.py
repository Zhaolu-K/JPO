import torch
import os
import json
from typing import List
import prompts
import math

BATCH_SIZE = 100
TEMPERATURE = 1.0
MAX_TOKENS = 850
MAX_TOKEN_FACT = 800

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN = 6144
TRUST_REMOTE_CODE = True
ENFORCE_EAGER = False
QUANTIZATION = "gptq_marlin"

SOURCE_PATH = '../data/jpo_dataset/train_sft.json'
SAVE_PATH = '../data/preprocessed_data/train_sft.json'
LAW_FILE_PATH = '../data/law.json'

with open(LAW_FILE_PATH, 'r', encoding='utf-8') as f:
    law = json.load(f)

def main():
    from vllm import LLM, SamplingParams

    print("Load model...")
    num_gpus = torch.cuda.device_count()
    print(num_gpus)

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=TRUST_REMOTE_CODE,
        enforce_eager=ENFORCE_EAGER,
        quantization=QUANTIZATION,
    )
    print("Load model completed.")

    def get_response(prompts: List[str]) -> List[str]:
        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        outputs = llm.generate(prompts, sampling_params)

        responses = []
        for output in outputs:
            generated_text = output.outputs[0].text
            responses.append(generated_text)

        if len(responses) > 0:
            print(responses[0] + f"\n{'=' * 30}\n")

        return responses

    all_data = []
    with open(SOURCE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                all_data.append(item)

    print(f"SFT dataset size: {len(all_data)}")

    cnt_batch = 0
    tot_batch = math.ceil(len(all_data) / BATCH_SIZE)

    for save_point in range(0, len(all_data), BATCH_SIZE * 8):
        previous_data = all_data[save_point: save_point + BATCH_SIZE * 8]
        data = []
        
        # build prompts
        prompts = []
        for item in previous_data:
            if len(item['fact']) <= MAX_TOKEN_FACT:
                data.append(item)
                relevant_articles = {i: law[i] for i in item["meta"]["relevant_articles"]}
                relevant_articles_str = ''
                cnt = 0
                for idx, content in relevant_articles.items():
                    cnt += 1
                    relevant_articles_str += f"({cnt}) 第{idx}条：{content}\n"

                accusation = item['meta']['accusation']
                accusation_str = ''
                for idx, content in enumerate(accusation):
                    accusation_str += f"({idx + 1}) {content}\n"

                prompt = f'''{prompts.INSTRUCTION_TEACHER_MODEL}

- 法律案件信息：{item["fact"]}


- 标准答案：
1、可参考的《刑法》法条：
{relevant_articles_str}
2、罪名：
{accusation_str}
3、判决刑期：{item["meta"]["term_of_imprisonment"]["imprisonment"]}个月

---

下面将根据要求生成推理过程。

## 推理过程：
'''
                prompts.append(prompt)

        # generate in batches
        responses = []
        for i in range(0, len(prompts), BATCH_SIZE):
            cnt_batch += 1
            batch_prompts = prompts[i: i + BATCH_SIZE]
            print(f"Processing batch {cnt_batch}/{tot_batch}. The batch size is {BATCH_SIZE} in current batch.")
            batch_responses = get_response(batch_prompts)
            responses.extend(batch_responses)

        # save result
        final_data = []
        for i in range(len(data)):
            final_data_item = {
                'fact': data[i]['fact'],
                'chain_of_thought': responses[i],
                'answer': json.dumps({
                    '法条': {art: law[art] for art in data[i]["meta"]["relevant_articles"]},
                    '罪名': data[i]['meta']['accusation'],
                    '刑期': data[i]['meta']['term_of_imprisonment']['imprisonment']
                }, ensure_ascii=False)
            }
            final_data.append(final_data_item)

        data_from_begin_to_current = []

        if os.path.exists(SAVE_PATH):
            with open(SAVE_PATH, 'r', encoding='utf-8') as f:
                data_from_begin_to_current = json.load(f)

        data_from_begin_to_current.extend(final_data)

        with open(SAVE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data_from_begin_to_current, f, ensure_ascii=False, indent=2)

        print(f"Finished! The result is saved to: {SAVE_PATH}")

if __name__ == '__main__':
    main()