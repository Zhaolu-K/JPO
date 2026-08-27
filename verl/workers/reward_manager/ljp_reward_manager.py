import statistics
from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score


class LJPRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        weight_per_token = []
        reward_per_seq = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            entropy_info = data_item.non_tensor_batch["entropy_info"]["entropy"]

            score, seq_weight_per_token = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                response_ids=valid_response_ids.cpu().numpy().tolist(),
                tokenizer=self.tokenizer,
                vocab_size=self.tokenizer.vocab_size,
                entropy_info=entropy_info,
                alpha=0.5,
                extra_info=extra_info,
            )

            weight_per_token.append(seq_weight_per_token)

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            # reward_tensor[i, valid_response_length - 1] = reward
            reward_per_seq.append(reward)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        # compute token level advantage
        seq_reward_mean = statistics.mean(reward_per_seq)
        seq_reward_std = statistics.pstdev(reward_per_seq)
        advantage_per_seq = [(reward - seq_reward_mean) / (seq_reward_std + 1e-8) for reward in reward_per_seq]

        for i in range(len(weight_per_token)):
            for j in range(len(weight_per_token[i])):
                reward_tensor[i, j] = advantage_per_seq[i] * self._compute_token_norm_weight(
                    weight_per_token[i][j], advantage_per_seq[i])

        data.batch['reward_per_seq'] = torch.tensor(reward_per_seq).to(data.batch['response_mask'].device)
        data.non_tensor_batch['advantage_per_seq'] = advantage_per_seq
        data.non_tensor_batch['weight_per_token'] = weight_per_token

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def _compute_token_norm_weight(
        self,
        token_weight: float,
        seq_advantage: float,
        beta: float = 0.6
    ):
        if seq_advantage > 0:
            return 1.0 + beta * token_weight
        elif seq_advantage < 0:
            return 1.0 - beta * token_weight
        else:
            return 1.0