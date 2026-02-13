from .ctx_manager import ContextManager
from .es_manager import EnvStateManager
from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import hydra
import os
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from .base_llm import ConcurrentLLM
# import time

import pdb
import json


class VllmWrapperWg: # Thi is a developing class for eval and test
	def __init__(self, config, tokenizer):
		self.config = config
		self.tokenizer = tokenizer
		model_name = config.actor_rollout_ref.model.path
		ro_config = config.actor_rollout_ref.rollout
		self.llm = LLM(
			model_name,
            enable_sleep_mode=True,
            tensor_parallel_size=ro_config.tensor_model_parallel_size,
            dtype=ro_config.dtype,
            enforce_eager=ro_config.enforce_eager,
            gpu_memory_utilization=ro_config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            # disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=ro_config.max_model_len,
            disable_log_stats=ro_config.disable_log_stats,
            max_num_batched_tokens=ro_config.max_num_batched_tokens,
            enable_chunked_prefill=ro_config.enable_chunked_prefill,
            enable_prefix_caching=True,
			trust_remote_code=True,
		)
		print("LLM initialized")
		self.sampling_params = SamplingParams(
			max_tokens=ro_config.response_length,
			temperature=ro_config.val_kwargs.temperature,
			top_p=ro_config.val_kwargs.top_p,
			top_k=ro_config.val_kwargs.top_k,
			# min_p=0.1,
		)

	def generate_sequences(self, lm_inputs: DataProto):
		"""
		Convert the input ids to text, and then generate the sequences. Finally create a dataproto. 
		This aligns with the verl Worker Group interface.
		"""
		# NOTE: free_cache_engine is not used in the vllm wrapper. Only used in the verl vllm.
		# cache_action = lm_inputs.meta_info.get('cache_action', None)
		input_ids = lm_inputs.batch['input_ids']
		input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
		# Check: 为什么要加下面这条？？加了输出都不对了。padding token不太对，不能移除
		# input_texts = [i.replace("<|endoftext|>", "") for i in input_texts] 
		# pdb.set_trace()
		outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
		texts = [output.outputs[0].text for output in outputs] 
		lm_outputs = DataProto()
		lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
		lm_outputs.meta_info = lm_inputs.meta_info

		return lm_outputs
	
class ApiCallingWrapperWg:
    """Wrapper class for API-based LLM calls that fits into the VERL framework"""
    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = model_info.generation_kwargs
        
        
        self.llm = ConcurrentLLM(
			provider=model_info.provider_name,
            model_name=model_info.model_name,
            max_concurrency=config.model_config.max_concurrency
        )
        
        print(f'API-based LLM ({model_info.provider_name} - {model_info.model_name}) initialized')


    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        """
        Convert the input ids to text, make API calls to generate responses, 
        and create a DataProto with the results.
        """

        messages_list = lm_inputs.non_tensor_batch['messages_list'].tolist()
        results, failed_messages = self.llm.run_batch(
            messages_list=messages_list,
            **self.llm_kwargs
        )
        assert not failed_messages, f"Failed to generate responses for the following messages: {failed_messages}"

        texts = [result["response"] for result in results]
        print(f'[DEBUG] texts: {texts}')
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
        lm_outputs.meta_info = lm_inputs.meta_info
        
        return lm_outputs

class LLMAgentProxy:
	"""
	The proxy means the llm agent is trying to generate some rollout **at this time**, **at this model state**, **at this env state from the env config**
	"""
	def __init__(self, config, actor_rollout_wg, tokenizer):
		self.config = config
		self.train_ctx_manager = ContextManager(config, tokenizer, mode="train")
		self.train_es_manager = EnvStateManager(config, mode="train")
		self.val_ctx_manager = ContextManager(config, tokenizer, mode="val")
		self.val_es_manager = EnvStateManager(config, mode="val")
		self.actor_wg = actor_rollout_wg
		self.tokenizer = tokenizer

	def generate_sequences(self, lm_inputs: DataProto):
		# TODO: add kv cache both for the vllm wrapper here and for verl vllm.
		if isinstance(self.actor_wg, RayWorkerGroup):
			padded_lm_inputs, pad_size = pad_dataproto_to_divisor(lm_inputs, self.actor_wg.world_size)
			padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs)
			lm_outputs = unpad_dataproto(padded_lm_outputs, pad_size=pad_size)
			lm_outputs.meta_info = lm_inputs.meta_info
			lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch
		elif isinstance(self.actor_wg, VllmWrapperWg) or isinstance(self.actor_wg, ApiCallingWrapperWg):
			lm_outputs = self.actor_wg.generate_sequences(lm_inputs)
		else:
			raise ValueError(f"Unsupported actor worker type: {type(self.actor_wg)}")

		return lm_outputs

	def rollout(self, dataproto: DataProto, val=False):
		es_manager = self.val_es_manager if val else self.train_es_manager
		ctx_manager = self.val_ctx_manager if val else self.train_ctx_manager
		print(json.dumps({
			"event": "rollout_reset_start",
			"val": val,
		}), flush=True)
		env_outputs = es_manager.reset()
		print(json.dumps({
			"event": "rollout_reset_done",
			"val": val,
			"env_outputs_len": len(env_outputs) if env_outputs is not None else None,
		}), flush=True)
		if env_outputs is None or len(env_outputs) == 0:
			print(json.dumps({
				"event": "rollout_empty_env_outputs_after_reset",
				"val": val,
			}), flush=True)
			rollout_states = es_manager.get_rollout_states()
			return ctx_manager.formulate_rollouts(rollout_states)
		batch_size = self.config.agent_proxy.lm_output_batch
		
		for i in range(self.config.agent_proxy.max_turn):
			print(f"Rollout step: {i}")
			print(json.dumps({
				"event": "rollout_get_lm_inputs_start",
				"step": i,
				"val": val,
				"env_outputs_len": len(env_outputs) if env_outputs is not None else None,
			}), flush=True)
			# pdb.set_trace()
			lm_inputs: DataProto = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
			try:
				lm_inputs_len = len(lm_inputs)
			except Exception:
				lm_inputs_len = None
			print(json.dumps({
				"event": "rollout_get_lm_inputs_done",
				"step": i,
				"val": val,
				"lm_inputs_len": lm_inputs_len,
				"meta_keys": sorted(list(dataproto.meta_info.keys())) if dataproto and dataproto.meta_info else None,
			}), flush=True)
			lm_inputs.meta_info = dataproto.meta_info # TODO: setup vllm early stop when max length is reached. make sure this can be done

			# lm_outputs: DataProto = self.generate_sequences(lm_inputs)
			total_len = len(lm_inputs)
			if total_len == 0:
				print(json.dumps({
					"event": "rollout_empty_lm_inputs",
					"step": i,
					"val": val,
					"env_outputs_len": len(env_outputs) if env_outputs is not None else None,
				}), flush=True)
				break
			chunks = (total_len + batch_size - 1) // batch_size
			print(json.dumps({
				"event": "rollout_chunking_start",
				"step": i,
				"val": val,
				"total_len": total_len,
				"batch_size": batch_size,
				"chunks": chunks,
			}), flush=True)
			sub_inputs_list = [lm_inputs.slice(start, min(start + batch_size, total_len)) for start in range(0, total_len, batch_size)]
			sub_outputs_list = []
			for chunk_idx, sub_inputs in enumerate(sub_inputs_list):
				print(json.dumps({
					"event": "rollout_chunk_generate_start",
					"step": i,
					"val": val,
					"chunk_idx": chunk_idx,
					"chunk_len": len(sub_inputs),
				}), flush=True)
				sub_outputs = self.generate_sequences(sub_inputs)
				print(json.dumps({
					"event": "rollout_chunk_generate_done",
					"step": i,
					"val": val,
					"chunk_idx": chunk_idx,
					"chunk_len": len(sub_inputs),
				}), flush=True)
				sub_outputs_list.append(sub_outputs)
			lm_outputs = DataProto.concat(sub_outputs_list)
			print(json.dumps({
				"event": "rollout_generate_done",
				"step": i,
				"val": val,
				"output_len": len(lm_outputs),
			}), flush=True)

			print(json.dumps({
				"event": "rollout_get_env_inputs_start",
				"step": i,
				"val": val,
				"output_len": len(lm_outputs),
			}), flush=True)
			env_inputs: List[Dict] = ctx_manager.get_env_inputs(lm_outputs)
			print(json.dumps({
				"event": "rollout_get_env_inputs_done",
				"step": i,
				"val": val,
				"env_inputs_len": len(env_inputs) if env_inputs is not None else None,
			}), flush=True)
			# pdb.set_trace()
			print(json.dumps({
				"event": "rollout_env_step_start",
				"step": i,
				"val": val,
				"env_inputs_len": len(env_inputs) if env_inputs is not None else None,
			}), flush=True)
			env_outputs: List[Dict] = es_manager.step(env_inputs)
			print(json.dumps({
				"event": "rollout_env_step_done",
				"step": i,
				"val": val,
				"env_outputs_len": len(env_outputs) if env_outputs is not None else None,
			}), flush=True)
			if len(env_outputs) == 0: # all finished
				break
		rollout_states = es_manager.get_rollout_states() 
		try:
			rollout_states_len = len(rollout_states)
		except Exception:
			rollout_states_len = None
		rollout_state_ids = []
		rollout_state_history_lens = []
		if rollout_states is not None:
			for state in rollout_states:
				rollout_state_ids.append(state.get("env_id"))
				history = state.get("history", []) if isinstance(state, dict) else []
				rollout_state_history_lens.append(len(history))
		print(json.dumps({
			"event": "rollout_states_summary",
			"val": val,
			"rollout_states_len": rollout_states_len,
			"rollout_state_ids": rollout_state_ids,
			"rollout_state_history_lens": rollout_state_history_lens,
		}), flush=True)
		rollouts = ctx_manager.formulate_rollouts(rollout_states)

		# pdb.set_trace()
		# rollouts.batch["rm_scores"], rollouts.batch["original_rm_scores"]
		# rollout_states[2]["history"][-1]['reward']
		# self.tokenizer.batch_decode(rollouts.batch['input_ids'], skip_special_tokens=False) # see all the trajectories

		return rollouts

@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
	# detect config name from python -m ragen.llm_agent.agent_proxy --config_name frozen_lake
	os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
	actor_wg = VllmWrapperWg(config, tokenizer)
	proxy = LLMAgentProxy(config, actor_wg, tokenizer)
	import time
	for _ in range(1):
		start_time = time.time()
		rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample':config.actor_rollout_ref.rollout.do_sample, 'validate': True}), val=True)
		end_time = time.time()
		print(f'rollout time: {end_time - start_time} seconds')
		# print rollout rewards from the rm_scores
		rm_scores = rollouts.batch["rm_scores"]
		metrics = rollouts.meta_info["metrics"]
		avg_reward = rm_scores.sum(-1).mean().item()
		print(f'rollout rewards: {avg_reward}')
		print(f'metrics:')
		for k, v in metrics.items():
			print(f'{k}: {v}')

		# 获取所有环境的对话历史
		rollout_states = proxy.val_es_manager.get_rollout_states()
		dialogue_histories = []
		for env in rollout_states:
			dialogue_histories.append({
				"env_id": env['env_id'],
				"dialogue_history": env.get('dialogue_history', []),
				"metrics": env.get('metrics', {})
			})

		# 保存到文件
		file_path = f"../RAGEN/run_logs/eval_dialogues/{config.model_path.split('/')[-1]}.json"
		with open(file_path, "w", encoding="utf-8") as f:
			json.dump(dialogue_histories, f, ensure_ascii=False, indent=2)
		print(f"Saved dialogue histories to {file_path}")
		# 保存指标
		metrics_file_path = f"../RAGEN/run_logs/eval_metrics/{config.model_path.split('/')[-1]}.json"
		with open(metrics_file_path, "w", encoding="utf-8") as f:
			json.dump(metrics, f, ensure_ascii=False, indent=2)
		print(f"Saved metrics to {metrics_file_path}")

# @hydra.main(version_base=None, config_path="../../config", config_name="evaluate_api_llm")
# def main(config):
# 	# detect config name from python -m ragen.llm_agent.agent_proxy --config_name frozen_lake
# 	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
# 	actor_wg = ApiCallingWrapperWg(config, tokenizer)
# 	proxy = LLMAgentProxy(config, actor_wg, tokenizer)
# 	import time
# 	start_time = time.time()
# 	rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': False, 'validate': True}), val=True)
# 	print(f'[DEBUG] rollouts: {rollouts}')
# 	end_time = time.time()
# 	print(f'rollout time: {end_time - start_time} seconds')
# 	# print rollout rewards from the rm_scores
# 	rm_scores = rollouts.batch["rm_scores"]
# 	metrics = rollouts.meta_info["metrics"]
# 	avg_reward = rm_scores.sum(-1).mean().item()
# 	print(f'rollout rewards: {avg_reward}')
# 	print(f'metrics:')
# 	for k, v in metrics.items():
# 		print(f'{k}: {v}')



if __name__ == "__main__":
	main()
