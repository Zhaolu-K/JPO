import json
from typing import List
import prompts

SOURCE_PATH = 'data/jpo_dataset/test.json'
SAVE_PATH_MODEL_PREDICT = 'output/qwen2_5_3b/qwen2_5_3b_predict.json'

MAX_TOKEN_FACT = 800

BATCH_SIZE = 100
TEMPERATURE = 1.0
MAX_TOKENS = 2048

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct" # or load your own trained model from a local path
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN = 6144
TRUST_REMOTE_CODE = True
ENFORCE_EAGER = False

def main():
    from vllm import LLM, SamplingParams

    print("Load model...")

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=TRUST_REMOTE_CODE,
        enforce_eager=ENFORCE_EAGER,
    )

    print("Load Model Completed.")

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

    test_dataset = []

    with open(SOURCE_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if len(data['fact']) <= MAX_TOKEN_FACT:
                data = json.loads(line.strip())
                test_dataset.append({
                    'fact': data['fact'],
                    'answer': {
                        '法条': data['meta']['relevant_articles'],
                        '罪名': data['meta']['accusation'],
                        '刑期': data['meta']['term_of_imprisonment']['imprisonment']
                    }
                })

    print(f"Test dataset size: {len(test_dataset)}")
    
    output = []

    for i in range(0, len(test_dataset), BATCH_SIZE):
        data = test_dataset[i: i + BATCH_SIZE]
        prompts = []
        for item in data:
            instruction = f'{prompts.INSTRUCTION_STUDENT_MODEL}\n\n{item["fact"]}'
            prompts.append(f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n")
        responses = get_response(llm, prompts)
        for item, response in zip(data, responses):
            output.append({
                'fact': item['fact'],
                'model_output': response,
                'answer': item['answer']
            })
    
    with open(SAVE_PATH_MODEL_PREDICT, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()