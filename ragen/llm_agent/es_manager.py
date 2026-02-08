"""
This is the environment state manager for the LLM agent.
author: Pingyue Zhang
date: 2025-03-30
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import PIL.Image
import hydra
import random
import numpy as np
import math
from openai import OpenAI
from ragen.llm_agent.vllm_local_client import VllmLocalClient

from tqdm import tqdm
import pdb
import json

from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from ragen.utils import register_resolvers
register_resolvers()

from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer


class OpenAIVLLMClient:
    def __init__(self, base_url, model, api_key="EMPTY"):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
        self.leave_out_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.juder_allowed_tokens = ["Yes", "No"] # Hardcoded for Harmbench classifier
        self.juder_allowed_token_ids = [self.tokenizer.encode(t, add_special_tokens=False)[0] for t in self.juder_allowed_tokens]
        self.juder_logit_bias = {token_id: 100 for token_id in self.juder_allowed_token_ids}

    def chat(self, messages, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
    
    def batch_complete(self, batch_messages, **kwargs):
        batch_converted_messages = []
        for messages in batch_messages:
            converted_messages = self.tokenizer.apply_chat_template(
                messages, tokenize=False
            )
            batch_converted_messages.append(converted_messages)

        response = self.client.completions.create(
            model=self.model,
            prompt=batch_converted_messages,
            **kwargs
        )

        return [choice.text.replace(self.leave_out_header, "") for choice in response.choices]

    def get_logprobs_chat(self, messages, index, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].logprobs.content[index].top_logprobs
    
    def batch_get_logprobs_complete(self, batch_prompts, index, **kwargs):
        response = self.client.completions.create(
            model=self.model,
            prompt=batch_prompts,
            logit_bias=self.juder_logit_bias,
            **kwargs
        )
        if index == 0:
            return [choice.logprobs.top_logprobs for choice in response.choices]


@dataclass
class EnvStatus:
    """Status of an environment"""
    truncated: bool = False # done but not success
    terminated: bool = False # done and success
    num_actions: int = 0 # current action step (single action)
    rewards: List[float] = field(default_factory=list) # rewards for each turn
    seed: Optional[int] = None # what seed is used to reset this environment



class EnvStateManager:
    """Manager for the environment state
    The class is responsible for managing multiple (kinds of) environments
    
    """
    def __init__(self, config, mode: str = "train"):
        self.sys_config = config
        self.mode = mode
        self.config = getattr(self.sys_config.es_manager, mode)
        self.env_groups = int(self.config.env_groups)
        self.group_size = self.config.group_size
        seed_cfg = getattr(self.sys_config, "seed", None)
        if seed_cfg is not None:
            self.base_seed = seed_cfg.get(mode, None)
        else:
            self.base_seed = None
        self.seed_counter = 0
        self.harmful_target_base_idx = 0
        from ragen.env.jailbreak.env import JailbreakEnv
        from ragen.env.jailbreak.config import JailbreakEnvConfig
        env_cfg = JailbreakEnvConfig()
        self.harmful_threshold = env_cfg.harmful_threshold
        self.targets = self._load_targets(env_cfg.data_paths)
        if self.mode == "train":
            # self.val_targets = self._load_targets([env_cfg.val_data_paths[1]]) if hasattr(env_cfg, 'val_data_paths') else []
            self.val_targets = self._load_targets(env_cfg.val_data_paths) if hasattr(env_cfg, 'val_data_paths') else []
        elif self.mode == "val":
            # self.val_targets = self._load_targets([env_cfg.val_data_paths[0]]) if hasattr(env_cfg, 'val_data_paths') else []
            self.val_targets = self._load_targets(env_cfg.val_data_paths) if hasattr(env_cfg, 'val_data_paths') else []
        
        self._init_envs()
        self.rollout_cache = None
        self._init_llms()

    def _init_llms(self):
        env_llm_cfg = self.sys_config.env_llm
        judger_llm_cfg = self.sys_config.judger_llm
        self.env_llm = OpenAIVLLMClient(
            base_url=env_llm_cfg.base_url,
            model=env_llm_cfg.model_path,
            api_key=getattr(env_llm_cfg, "api_key", "EMPTY")
        )
        self.judger_llm = OpenAIVLLMClient(
            base_url=judger_llm_cfg.base_url,
            model=judger_llm_cfg.model_path,
            api_key=getattr(judger_llm_cfg, "api_key", "EMPTY")
        )
        self.env_llm_params = {
            "max_tokens": getattr(env_llm_cfg, "max_tokens", 4096),
            "temperature": getattr(env_llm_cfg, "temperature", 0.7),
            "stop": getattr(env_llm_cfg, "stop", None)
        }
        self.judger_llm_params = {
            "max_tokens": getattr(judger_llm_cfg, "max_tokens", 1),
            "temperature": getattr(judger_llm_cfg, "temperature", 0.0),
            "logprobs": 2,
        }

    def _chat_batch(self, messages_batch, llm_params):
        results = [None] * len(messages_batch)
        with ThreadPoolExecutor(max_workers=256) as executor:
            futures = {
                executor.submit(self.env_llm.chat, messages, **llm_params): idx
                for idx, messages in enumerate(messages_batch)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error in chat_batch for idx {idx}: {e}")
                    results[idx] = ""
        return results

    def _chat_batch_allinone(self, messages_batch, llm_params):
        results = self.env_llm.batch_complete(messages_batch, **llm_params)
        # pdb.set_trace()
        return results
        

    def _get_logprobs_batch_multi_thread(self, messages_batch, llm_params):
        results = [None] * len(messages_batch)
        with ThreadPoolExecutor(max_workers=256) as executor:
            futures = {
                executor.submit(self.judger_llm.get_logprobs, messages, **llm_params): idx
                for idx, messages in enumerate(messages_batch)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error in get_logprobs_batch for idx {idx}: {e}")
                    results[idx] = None
        return results

    def _get_logprobs_batch(self, messages_batch, llm_params):
        index = int(self.sys_config.judger_llm.max_tokens) - 1
        results = self.judger_llm.batch_get_logprobs_complete(batch_prompts=messages_batch, index=index, **llm_params)
        return results

    def _load_targets(self, data_paths):
        import pandas as pd
        all_targets = []
        for data_path in data_paths:
            try:
                if data_path.endswith('.parquet'):
                    df = pd.read_parquet(data_path)
                    targets = df['prompt'].dropna().tolist()
                    print(data_path, "data length: ", len(targets))
                    all_targets.extend(targets)
                elif 'strongreject' in data_path and data_path.endswith('.csv'):
                    strongreject_df = pd.read_csv(data_path)
                    targets = strongreject_df[strongreject_df["source"] != "AdvBench"]["forbidden_prompt"].tolist()
                    print(data_path, "data length: ", len(targets))
                    all_targets.extend(targets)
                elif 'jbb' in data_path and  data_path.endswith('.csv'):
                    jbb_df = pd.read_csv(data_path)
                    targets = jbb_df[jbb_df["Source"] == "Original"]["Goal"].tolist()
                    all_targets.extend(targets)
                elif 'harmbench' in data_path and data_path.endswith('.csv'):
                    harmbench_df = pd.read_csv(data_path)
                    targets = harmbench_df[harmbench_df["FunctionalCategory"] == "standard"]["Behavior"].tolist()
                    all_targets.extend(targets)
                elif 'advbench' in data_path and data_path.endswith('.csv'):
                    advbench_df = pd.read_csv(data_path)
                    targets = advbench_df["instruct"].tolist()
                    all_targets.extend(targets)
            except Exception as e:
                print(f"Warning: Failed to load data from {data_path}: {e}")
                continue
        return all_targets

    def _init_envs(self):
        """Initialize the environments. train_envs and val_envs are lists of envs:
        Input: tags: ["SimpleSokoban", "HarderSokoban"]; n_groups: [1, 1]; group_size: 16
        Output: envs: List[Dict], each **entry** is a dict with keys: tag, group_id, env_id, env, env_config, status
        Example: [{"tag": "SimpleSokoban", "group_id": 0, "env_id": 0, "env": env, "config": env_config, "status": EnvStatus()},
            ...
            {"tag": "SimpleSokoban", "group_id": 0, "env_id": 15 (group_size - 1), ...},
            {"tag": "HarderSokoban", "group_id": 1, "env_id": 16, ...}
            ...]
        """
        assert sum(self.config.env_configs.n_groups) == self.env_groups, f"Sum of n_groups must equal env_groups. Got sum({self.config.env_configs.n_groups}) != {self.env_groups}"
        assert len(self.config.env_configs.tags) == len(self.config.env_configs.n_groups), f"Number of tags must equal number of n_groups. Got {len(self.config.env_configs.tags)} != {len(self.config.env_configs.n_groups)}"
        self.envs = self._init_env_instances(self.config)

    def _init_env_instances(self, config):
        env_list = []
        done_groups = 0
        for tag, n_group in zip(config.env_configs.tags, config.env_configs.n_groups):
            for env_id in range(done_groups * self.group_size, (done_groups + n_group) * self.group_size):
                cfg_template = self.sys_config.custom_envs[tag]
                env_class = cfg_template.env_type
                max_actions_per_traj = cfg_template.max_actions_per_traj
                if cfg_template.env_config is None:
                    env_config = REGISTERED_ENV_CONFIGS[env_class]()
                else:
                    env_config = REGISTERED_ENV_CONFIGS[env_class](**cfg_template.env_config)
                env_obj = REGISTERED_ENVS[env_class](env_config)
                entry = {'tag': tag, 'group_id': env_id // self.group_size, 'env_id': env_id, 
                        'env': env_obj, 'config': env_config, 'status': EnvStatus(), 'max_actions_per_traj': max_actions_per_traj}
                env_list.append(entry)
            done_groups += n_group
        return env_list

    def reset(self, seed: Optional[int] = None):
        """
        Reset the environments and get initial observation
        build up rollout cache like [{"env_id": int, "history": List[Dict], "group_id": int}, ...]
        """
        print(json.dumps({
            "event": "es_manager_reset_start",
            "mode": self.mode,
            "seed_arg": seed,
            "env_groups": self.env_groups,
            "group_size": self.group_size,
            "num_envs": len(self.envs) if hasattr(self, "envs") else None,
        }), flush=True)
        def _expand_seed(seed: int):
            seeds = [[seed + i] * self.group_size for i in range(self.env_groups)] # [[seed, ..., seed], [seed+1, ..., seed+1], ...]
            return sum(seeds, [])

        envs = self.envs
        rollout_cache = [{"env_id": entry['env_id'], "history": [], "group_id": entry['group_id'], "tag": entry['tag'], "penalty": 0} for entry in envs]

        # reset all environments
        if seed is None:
            if self.mode == "train":
                if self.base_seed is not None:
                    seed = int(self.base_seed + self.seed_counter)
                    self.seed_counter += self.env_groups
                else:
                    seed = int(random.randint(0, 1000000))
            else:
                seed = int(123 if self.base_seed is None else self.base_seed)
        else:
            if self.mode == "train" and self.base_seed is not None:
                self.seed_counter = int(seed - self.base_seed + 1)
            seed = int(seed)
        seeds = _expand_seed(seed) if envs else []

        if self.mode == "val":
            total_targets = len(self.val_targets)
        else:
            total_targets = len(self.targets)

        for entry, seed in zip(envs, seeds):
            group_id = entry['group_id']
            if self.mode == "val":
                harmful_target = self.val_targets[(self.harmful_target_base_idx + group_id) % total_targets]
            else:
                harmful_target = self.targets[(self.harmful_target_base_idx + group_id) % total_targets]
            print(json.dumps({
                "event": "env_reset_start",
                "mode": self.mode,
                "env_id": entry['env_id'],
                "group_id": group_id,
                "tag": entry.get('tag'),
                "seed": seed,
            }), flush=True)
            entry['env'].reset(seed=seed, mode=self.mode, harmful_target=harmful_target)
            entry['status'] = EnvStatus(seed=seed)
            print(json.dumps({
                "event": "env_reset_done",
                "mode": self.mode,
                "env_id": entry['env_id'],
                "group_id": group_id,
                "tag": entry.get('tag'),
                "seed": seed,
            }), flush=True)
            
        for cache in rollout_cache:
            cache['init_prompt'] = envs[cache['env_id']]['env'].init_prompt
            cache['harmful_target'] = envs[cache['env_id']]['env'].current_target

        # update rollout cache
        for cache, env in zip(rollout_cache, envs):
            next_state = self._handle_mm_state(env['env'].render())
            cache['history'] = self._update_cache_history(cache['history'], next_state=next_state, actions_left=env['max_actions_per_traj'], num_actions_info=None)
            
        if total_targets > 0:
            self.harmful_target_base_idx = (self.harmful_target_base_idx + self.env_groups) % total_targets
        else:
            self.harmful_target_base_idx = 0
        self.rollout_cache = rollout_cache if rollout_cache else []
        print(json.dumps({
            "event": "es_manager_reset_done",
            "mode": self.mode,
            "seed": seed,
            "rollout_cache_len": len(self.rollout_cache),
        }), flush=True)
        # pdb.set_trace()
        return self.rollout_cache

    def step(self, all_env_inputs: List[Dict]):
        """Step the environments.
        1. extract valid actions from the action lookup table (if exists) and execute the actions, and update rollout cache
        2. Since rollout does not need to act over done envs, whenever the environment is done, we only update rollout cache, but not output env_outputs.
        Input:
        all_env_inputs: List[Dict]
            {env_id: int, llm_response: str, actions: List[str]}
            NOTE: should use env_id as index for existing some already done envs
        env_outputs: List[Dict]
            {env_id: int, history: List[Dict][{state: str, actions: List[str], reward: float, info: Dict, llm_response: str, llm_raw_response: str, (Optional)images: List[PIL.Image.Image]}]}
        """
        def _execute_actions(env, actions):
            acc_reward, turn_info, turn_done = 0, {}, False
            executed_actions = []
            for action in actions:
                _, reward, done, info = env.step(action)
                acc_reward += reward
                turn_info.update(info) # NOTE: currently use last info for multi-action
                executed_actions.append(action)
                if done:
                    turn_done = True
                    break
            
            return acc_reward, turn_info, turn_done, executed_actions

        def _log_env_state(status, history, cur_obs, max_actions_per_traj, executed_actions, all_actions, acc_reward, turn_done, turn_info, env_input, env_response):
            obs = self._handle_mm_state(cur_obs)
            status.num_actions += len(executed_actions)
            status.rewards.append(acc_reward) # NOTE use turn-wise acc_reward
            actions_left = max_actions_per_traj - status.num_actions
            if turn_done:
                # status.terminated = True # TODO check terminated definition in gymnasium
                # status.truncated = not turn_info.get('success', False)
                status.truncated = not turn_info.get('turn_success', False) # TODO check truncated definition in gymnasium
                status.terminated = turn_info.get('turn_success', False)
            history = self._update_cache_history(history, next_state=obs, actions_left=actions_left, num_actions_info={
                'actions': executed_actions, 'reward': acc_reward, 'info': turn_info,
                'llm_response': env_input['llm_response'], 'llm_raw_response': env_input['llm_raw_response'],
                'env_response': env_response
            })
            # filter out invalid actions
            # history = [content for content in history[:-1] if content['actions']] + [history[-1]]
            return status, history

        envs = self.envs
        env_outputs = []

        env_llm_batch = []
        env_llm_indices = []
        
        for env_input in all_env_inputs:
            entry = envs[env_input['env_id']]
            env_id, env = entry['env_id'], entry['env']
            
            if env.done:
                continue
                
            env_messages = env.get_env_llm_messages(env_input['actions'][0])
            if env_messages:
                env_llm_batch.append(env_messages)
                env_llm_indices.append(env_id)

        env_responses = {}
        if env_llm_batch:
            try:
                env_llm_responses = self._chat_batch(env_llm_batch, self.env_llm_params)
                for env_id, response in zip(env_llm_indices, env_llm_responses):
                    env_responses[env_id] = response
            except Exception as e:
                print(f"Error in batch env LLM call: {e}")
                for env_id in env_llm_indices:
                    env_responses[env_id] = ""

        judger_llm_batch = []
        judger_llm_indices = []
        
        for env_input in all_env_inputs:
            entry = envs[env_input['env_id']]
            env_id, env = entry['env_id'], entry['env']
            
            if env.done:
                continue
                
            env.dialogue_history.append({"role": "assistant", "content": env_input['actions'][0]})
            if env_id in env_responses:
                env.dialogue_history.append({"role": "user", "content": env_responses[env_id]})
            
            judger_messages = env.get_judger_llm_messages()
            if judger_messages:
                judger_llm_batch.append(judger_messages)
                judger_llm_indices.append(env_id)

        judger_scores = {}
        if judger_llm_batch:
            try:
                judger_responses = self._get_logprobs_batch(judger_llm_batch, self.judger_llm_params)
                for env_id, logprobs in zip(judger_llm_indices, judger_responses):
                    try:
                        if logprobs and isinstance(logprobs, list):
                            unsafe_logprob = logprobs[0][self.sys_config.judger_llm.wanted_token]
                            if unsafe_logprob is not None:
                                if math.exp(unsafe_logprob) > 1:
                                    raise ValueError(f"Unsafe prob is too large: {math.exp(unsafe_logprob)}. Unsafe logprob: {unsafe_logprob}")
                                judger_scores[env_id] = math.exp(unsafe_logprob)
                            else:
                                print(f"Warning: No unsafe logprob found for env {env_id}")
                                judger_scores[env_id] = 0.0
                        else:
                            print(f"Warning: logprobs is not a list for env {env_id}")
                            judger_scores[env_id] = 0.0
                    except Exception as e:
                        if str(e) == "'Yes'":
                            judger_scores[env_id] = 0.0 # Too small even set logit_bias
                        else:
                            print(f"Error parsing logprobs for env {env_id}: {e}")
                            judger_scores[env_id] = 0.0
            except Exception as e:
                print(f"Error in batch judger LLM call: {e}")
                for env_id in judger_llm_indices:
                    judger_scores[env_id] = 0.0

        for env_input in all_env_inputs:
            entry = envs[env_input['env_id']]
            env_id, env = entry['env_id'], entry['env']
            actions_left_before = entry['max_actions_per_traj'] - entry['status'].num_actions

            if env.done:
                continue

            env_response = env_responses.get(env_id, "")
            score = judger_scores.get(env_id, 0.0)
            
            _, reward, done, info = env.step_with_llm_response(
                env_input['actions'][0], env_response, score
            )
            
            valid_actions = self._extract_map_valid_actions(entry, env_input['actions'])
            if len(valid_actions) > 1:
                raise ValueError("Multi-action not supported, a query is seen as an action")
                acc_reward, turn_info, turn_done, executed_actions = _execute_actions(env, valid_actions[1:])
                reward += acc_reward
                info.update(turn_info)
                if turn_done:
                    done = True
            else:
                executed_actions = valid_actions[:1] if valid_actions else []
                turn_info = info
            
            if len(valid_actions) != len(env_input['actions']) or not valid_actions:
                self.rollout_cache[env_id]["penalty"] += self.sys_config.es_manager.format_penalty
                
            status, history = _log_env_state(
                entry['status'],
                self.rollout_cache[env_id]['history'],
                env.render(),
                entry['max_actions_per_traj'],
                executed_actions,
                valid_actions,
                reward,
                done,
                turn_info,
                env_input,
                env_response
            )
            entry['status'] = status
            if entry['status'].num_actions >= entry['max_actions_per_traj'] and not done:
                entry['status'].truncated = True
                entry['status'].terminated = False
                done = True
            self.rollout_cache[env_id]['history'] = history
            self.rollout_cache[env_id]['dialogue_history'] = env.dialogue_history
            self.rollout_cache[env_id]['turn_scores'] = [turn['info']['score'] for turn in history[:-1]]
            if not done: # NOTE done environments are not sent for further llm generation (for efficiency)
                env_outputs.append(self.rollout_cache[env_id])

        return env_outputs

    def get_rollout_states(self):
        """Get the final output for all environment"""
        envs = self.envs
        rollout_cache = self.rollout_cache
        TURN_LVL_METRICS = ['action_is_effective', 'action_is_valid', 'end_of_page']

        # add metrics to rollout cache
        for entry, cache in zip(envs, rollout_cache):
            status = entry['status']
            env_metric = {
                'success': float(status.terminated and (not status.truncated)),
                'num_actions': status.num_actions,
            }
            custom_metric = {}
            for turn in cache['history']:
                for k, v in turn.get('info', {}).items():
                    if k not in custom_metric:
                        custom_metric[k] = []
                    custom_metric[k].append(v)
            for k, v in custom_metric.items():
                env_metric[k] = v


            cache['history'][-1]['metrics'] = custom_metric
            env_metric = {f"{entry['tag']}/{k}": v for k, v in env_metric.items()}
            cache['metrics'] = env_metric
            if entry['tag'] == "MetamathQA":
                cache['correct_answer'] = entry['env'].correct_answer

        # calculate pass@k where k is the group size
        group_success = {}
        for entry, cache in zip(envs, rollout_cache):
            key = (entry['tag'], entry['group_id'])
            success_val = cache['metrics'].get(f"{entry['tag']}/success", 0.0)
            group_success.setdefault(key, []).append(success_val)

        for (tag, gid), succ_list in group_success.items():
            pass_success = float(any(succ_list))
            for entry, cache in zip(envs, rollout_cache):
                if entry['tag'] == tag and entry['group_id'] == gid:
                    cache['metrics'][f"{tag}/pass@{self.group_size}"] = pass_success
        return rollout_cache




    def _update_cache_history(self, history: List[Dict], next_state, actions_left, num_actions_info: Optional[Dict] = None):
        """
        Update last step info and append state to history
        """
        if num_actions_info is not None: # update last step info
            assert len(history), "History should not be empty"
            history[-1].update(num_actions_info)
        
        entry = {} # append state to history
        if isinstance(next_state, str): # text state
            entry['state'] = next_state
        else: # multimodal state
            entry['state'] = "<images>" * len(next_state)
            entry['images'] = next_state
        entry['actions_left'] = actions_left
        history.append(entry)
        return history

    def _extract_map_valid_actions(self, entry: Dict, actions: List[str]):
        """extract valid actions from the action lookup table (if exists)"""
        mapped_actions = []
        action_lookup = getattr(entry['env'].config, 'action_lookup', None)
        if action_lookup is None:
            mapped_actions = actions
        else: # the envs have pre-defined action lookup
            rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
            actions = [action.lower() for action in actions]
            mapped_actions = [rev_action_lookup[action] for action in actions if action in rev_action_lookup]
        return mapped_actions
    
    def _handle_mm_state(self, state: Union[str, np.ndarray, list[np.ndarray]]):
        """Handle the state from the environment
        """
        if isinstance(state, str): # text state
            return state
        elif isinstance(state, np.ndarray): # when env state is a single image, convert it to a list to unify output format
            state = [state]
        results = [PIL.Image.fromarray(_state, mode='RGB') for _state in state]
        return results
        
    def render(self):
        rendered_list = [entry['env'].render() for entry in self.envs]
        return rendered_list

    def close(self):
        for entry in self.envs:
            entry['env'].close()




@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
    """
    Unit test for EnvStateManager
    """
    es_manager = EnvStateManager(config, mode="train")
    print("Initializing environments...")
    es_manager.reset(seed=123)

    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRunning step for training environments...")
    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")

    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go left, go up",
            "llm_response": "Go left, go up",
            "actions": ["left", "up"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go up, go up",
            "llm_response": "Go up, go up",
            "actions": ["up", "up", "up", "up", "up"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRendering final output...")
    final_outputs = es_manager.get_rollout_states()
    print(f"final outputs[:4]: {final_outputs[:4]}")
    
    print("\nClosing environments...")
    es_manager.close()
    print("Test completed successfully!")


if __name__ == "__main__":
	main()
