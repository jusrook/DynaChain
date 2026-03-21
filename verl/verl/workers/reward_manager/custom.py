from collections import defaultdict
import concurrent.futures
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("custom")
class CustomRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        max_workers=42  # 添加最大线程数参数
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.max_workers = max_workers  # 保存最大线程数

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def _process_item(self, i, data_item):
        """处理单个数据项的辅助函数"""
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        if 'prompt' in data_item.non_tensor_batch:
            prompt_str = data_item.non_tensor_batch['prompt'][0]['content']
        else:
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        eos_token = self.tokenizer.eos_token
        if response_str.endswith(eos_token):
            response_str = response_str[: -len(eos_token)]

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data_item.non_tensor_batch[self.reward_fn_key]
        extra_info = data_item.non_tensor_batch.get("extra_info", None)

        result = self.compute_score(
            query=prompt_str,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        score = result["score"] if isinstance(result, dict) else result
        reward = score

        if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
            overlong_buffer_len = self.overlong_buffer_cfg.len
            expected_len = self.max_resp_len - overlong_buffer_len
            exceed_len = valid_response_length - expected_len
            overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            reward += overlong_reward

        # 准备返回结果
        result_dict = {
            "index": i,
            "reward": reward,
            "valid_response_length": valid_response_length,
            "data_source": data_source,
            "print_info": {
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "result": result,
                "score": score
            }
        }

        # 如果是字典结果，添加额外信息
        if isinstance(result, dict):
            result_dict["reward_extra_info"] = result
        else:
            result_dict["reward_extra_info"] = {"acc": score}

        # 处理过长缓冲区的额外信息
        if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
            if self.overlong_buffer_cfg.log:
                result_dict["reward_extra_info"]["overlong_reward"] = overlong_reward
                result_dict["reward_extra_info"]["overlong"] = overlong_reward < 0

        return result_dict

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(lambda: [0.0] * len(data))
        already_print_data_sources = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(len(data)):
                data_item = data[i]
                futures.append(executor.submit(self._process_item, i, data_item))

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                i = result["index"]
                
                # 设置奖励张量
                reward_tensor[i, result["valid_response_length"] - 1] = result["reward"]
                
                # 收集额外信息
                for key, value in result["reward_extra_info"].items():
                    reward_extra_info[key][i] = value
                
                # 处理打印
                data_source = result["data_source"]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print_info = result["print_info"]
                    print("[prompt]", print_info["prompt_str"])
                    print("[response]", print_info["response_str"])
                    print("[ground_truth]", print_info["ground_truth"])
                    if isinstance(print_info["result"], dict):
                        for key, value in print_info["result"].items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", print_info["score"])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor