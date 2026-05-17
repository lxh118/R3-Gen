"""
API服务配置通过环境变量：
- EDIT_SERVER_ENDPOINTS: 图像编辑服务端点（逗号分隔）
- REWARD_SERVER_ENDPOINTS: 奖励服务端点（逗号分隔）
"""

import re
import json
import os
import sys
from typing import Any, Optional, Dict, Tuple
from collections import Counter
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# 路径配置：从 examples/reward_function/ 向上回退并指向 distributed_services
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)  # examples/
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)  # project root/
DISTRIBUTED_SERVICES_DIR = os.path.join(GRANDPARENT_DIR, "distributed_services")
if DISTRIBUTED_SERVICES_DIR not in sys.path:
    sys.path.insert(0, DISTRIBUTED_SERVICES_DIR)
    
try:
    from clients.api_client import (
        ImageEditClient,
        RewardClient,
        create_clients_from_env,
    )
    API_CLIENT_AVAILABLE = True
except ImportError:
    API_CLIENT_AVAILABLE = False
    print("Warning: API client not available. Please check distributed_services/clients/api_client.py")

# 全局客户端实例（延迟初始化）
_edit_client: Optional[ImageEditClient] = None
_reward_client: Optional[RewardClient] = None
_sam3_reward_client: Optional[RewardClient] = None

# 日志控制标志（避免重复打印）
_client_warning_printed: Dict[str, bool] = {
    "edit_client": False,
    "reward_client": False,
    "sam3_reward_client": False,
    "max_workers": False,
    "omniverifier_client_error": False,
    "clip_client_error": False,
    "omniverifier_endpoints_empty": False,
    "reward_type_fallback": False,
    "default_reward_type": False,
}


def _get_env_int(key: str, default: int) -> int:
    """安全地从环境变量获取整数值"""
    value = os.environ.get(key, str(default))
    value = value.strip().strip('"').strip("'")
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _get_env_bool(key: str, default: bool = True) -> bool:
    """安全地从环境变量获取布尔值"""
    value = os.environ.get(key, str(default).lower())
    value = value.strip().strip('"').strip("'").lower()
    return value == "true"


# Category到reward_type的映射
_CATEGORY_TO_REWARD_TYPE = {
    "color": "clip",
    "shape": "clip",
    "texture": "clip",
    "spatial": "clip",
    "numeracy": "clip",
    "object": "clip",
    "complex": "clip",
    "non": "clip",
}

# 支持的奖励类型
_SUPPORTED_REWARD_TYPES = {"clip", "omniverifier", "sam3", "mixed"}


def _get_reward_type_for_current_gpu() -> Optional[str]:
    """根据 REWARD_TYPE_PER_GPU 环境变量和当前 GPU ID 获取奖励类型"""
    reward_type_per_gpu = os.environ.get("REWARD_TYPE_PER_GPU", "")
    if not reward_type_per_gpu:
        return None
    
    gpu_id = 0
    local_rank = os.environ.get("LOCAL_RANK", None)
    if local_rank is not None:
        try:
            gpu_id = int(local_rank)
        except (ValueError, TypeError):
            gpu_id = 0
    
    try:
        mappings = {}
        for mapping in reward_type_per_gpu.split(","):
            mapping = mapping.strip()
            if ":" in mapping:
                gpu_str, reward_type = mapping.split(":", 1)
                try:
                    gpu_num = int(gpu_str.strip())
                    mappings[gpu_num] = reward_type.strip()
                except (ValueError, TypeError):
                    continue
        return mappings.get(gpu_id)
    except Exception:
        return None


def initialize_clients():
    """初始化API客户端"""
    global _edit_client, _reward_client, _sam3_reward_client, _client_warning_printed
    
    if not API_CLIENT_AVAILABLE:
        raise RuntimeError("API client not available. Cannot initialize clients.")
    
    # 检查重载配置
    if any([_edit_client, _reward_client, _sam3_reward_client]):
        try:
            from clients.api_client import _check_and_reload_config
            if _check_and_reload_config(_edit_client, _reward_client, _sam3_reward_client):
                print("[INFO] 检测到配置文件修改，已自动更新客户端端点")
        except Exception:
            pass
    
    if any(c is None for c in [_edit_client, _reward_client, _sam3_reward_client]):
        _edit_client, _reward_client, _sam3_reward_client = create_clients_from_env()
    
    # 打印警告
    if _edit_client is None and not _client_warning_printed.get("edit_client", False):
        print("Warning: Image edit client not initialized. Check EDIT_SERVER_ENDPOINTS environment variable.")
        _client_warning_printed["edit_client"] = True
    
    if _reward_client is None and not _client_warning_printed.get("reward_client", False):
        print("Warning: Reward client not initialized. Check REWARD_SERVER_ENDPOINTS environment variable.")
        _client_warning_printed["reward_client"] = True
        
    return _edit_client, _reward_client, _sam3_reward_client


def get_reward_client_for_type(reward_type: str):
    """根据奖励类型获取正确的奖励客户端"""
    global _reward_client, _sam3_reward_client, _client_warning_printed
    
    if reward_type == "sam3":
        return _sam3_reward_client
    elif reward_type == "omniverifier":
        omniverifier_endpoints = os.environ.get("OMNIVERIFIER_REWARD_SERVER_ENDPOINTS", "")
        if omniverifier_endpoints:
            try:
                from clients.api_client import RewardClient
                endpoints = [e.strip() for e in omniverifier_endpoints.split(",") if e.strip()]
                if endpoints:
                    return RewardClient(
                        endpoints,
                        timeout=_get_env_int("API_REQUEST_TIMEOUT", 180),
                        max_retries=_get_env_int("API_MAX_RETRIES", 3),
                        health_check_timeout=_get_env_int("API_HEALTH_CHECK_TIMEOUT", 5),
                        health_check_interval=_get_env_int("API_HEALTH_CHECK_INTERVAL", 600),
                        enable_health_check=_get_env_bool("API_ENABLE_HEALTH_CHECK", True)
                    )
            except Exception as e:
                if not _client_warning_printed.get("omniverifier_client_error", False):
                    print(f"[WARNING] 无法创建 OmniVerifier 客户端: {e}")
                    _client_warning_printed["omniverifier_client_error"] = True
        
        env_reward_type = os.environ.get("REWARD_TYPE", "").lower()
        if env_reward_type in ["mixed", "clip"]:
            if not _client_warning_printed.get("omniverifier_endpoints_empty", False):
                print(f"[WARNING] 请求 omniverifier 但端点为空，且 REWARD_TYPE={env_reward_type}")
                _client_warning_printed["omniverifier_endpoints_empty"] = True
            return None
        
        if env_reward_type == "omniverifier" and not _client_warning_printed.get("reward_type_fallback", False):
            print(f"[WARNING] REWARD_TYPE=omniverifier 但专用端点为空，回退到默认 REWARD_SERVER_ENDPOINTS")
            _client_warning_printed["reward_type_fallback"] = True
        
        return _reward_client
    else:
        # Default / CLIP
        clip_endpoints = os.environ.get("CLIP_REWARD_SERVER_ENDPOINTS", "")
        if clip_endpoints:
            try:
                from clients.api_client import RewardClient
                endpoints = [e.strip() for e in clip_endpoints.split(",") if e.strip()]
                if endpoints:
                    return RewardClient(
                        endpoints,
                        timeout=_get_env_int("API_REQUEST_TIMEOUT", 180),
                        max_retries=_get_env_int("API_MAX_RETRIES", 3),
                        health_check_timeout=_get_env_int("API_HEALTH_CHECK_TIMEOUT", 5),
                        health_check_interval=_get_env_int("API_HEALTH_CHECK_INTERVAL", 600),
                        enable_health_check=_get_env_bool("API_ENABLE_HEALTH_CHECK", True)
                    )
            except Exception as e:
                if not _client_warning_printed.get("clip_client_error", False):
                    print(f"[WARNING] 无法创建 CLIP 客户端: {e}")
                    _client_warning_printed["clip_client_error"] = True
        return _reward_client


def filter_thinking_part(response, eos_token=None):
    """
    提取answer部分（去除think标签）
    支持标准格式 <thinking>...</thinking> 和 仅结束标签格式
    """
    response_start = 0
    success = False
    think_tag_start = '<thinking>'
    think_tag_end = '</thinking>'
    
    think_end = response.rfind(think_tag_end)
    
    if think_end != -1:
        response_start = think_end + len(think_tag_end)
        success = True
    else:
        # 兼容性尝试：如果找不到结束标签，尝试找开始标签（虽然这在think格式检查中会得0分）
        think_start = response.find(think_tag_start)
        if think_start != -1:
            response_start = think_start + len(think_tag_start)
            success = True
    
    if eos_token is not None:
        response_end = response.find(eos_token, response_start)
    else:
        response_end = len(response)
    response = response[response_start:response_end]
    return response, success


def think_format_reward(response: str) -> float:
    """检查think标签格式奖励"""
    r = (response or "").strip()
    think_tag_start = "<thinking>"
    think_tag_end = "</thinking>"
    
    think_end = r.rfind(think_tag_end)
    if think_end == -1:
        return 0.0
    
    think_start = r.find(think_tag_start, 0, think_end)
    # 支持两种格式：完整的 <thinking>...</thinking> 或者只有 </thinking> 前面的内容
    if think_start != -1:
        think = r[think_start + len(think_tag_start):think_end]
    else:
        think = r[:think_end]
    
    ans = r[think_end + len(think_tag_end):].strip()
    
    # 关键检查：必须有非空的思考内容，非空的答案，且答案不应该包含嵌套标签
    if think.strip() and ans.strip() and think_tag_start not in ans:
        return 1.0
    return 0.0


def is_valid_edit_prompt(edit_prompt: str) -> bool:
    """检查edit_prompt是否有效"""
    if not edit_prompt:
        return False
    
    edit_prompt_clean = edit_prompt.strip()
    if not edit_prompt_clean:
        return False
    
    # 长度检查
    if len(edit_prompt_clean) <= 10:
        return False
    
    # 无效词检查
    edit_prompt_lower = edit_prompt_clean.lower()
    if edit_prompt_lower in ["remain unchanged", "no edit", ""]:
        return False
    
    # 模板检查
    template_patterns = [
        "a concrete, location-specific editing instruction",
        "concrete, location-specific editing instruction to fix the error",
        "provide a concrete, location-specific editing instruction",
        "location-specific editing instruction",
    ]
    for pattern in template_patterns:
        if pattern in edit_prompt_lower:
            return False
    
    # 动作词检查
    action_words = ['add', 'remove', 'replace', 'change', 'move', 'delete', 'place', 'position', 'shift', 'make', 'modify', 'update']
    has_action_word = any(word in edit_prompt_lower for word in action_words)
    
    if not has_action_word and len(edit_prompt_clean) < 20:
        return False
    
    return True


def check_format_collapse(text: str, min_words: int = 5, max_consecutive_repeat: int = 5) -> bool:
    """检测文本格式崩溃"""
    if not text:
        return False
    
    # 检测字符重复 (如 " " " " " ...)
    if len(text) > 10:
        consecutive_char_count = 1
        max_char_repeat = 1
        for i in range(1, len(text)):
            if text[i] == text[i-1] and text[i] in [' ', '"', "'", '\n', '\t']:
                consecutive_char_count += 1
                max_char_repeat = max(max_char_repeat, consecutive_char_count)
            else:
                consecutive_char_count = 1
        if max_char_repeat > 20:
            return True
    
    # 检测单词重复
    words = text.split()
    if len(words) <= min_words:
        return False
    
    consecutive_repeat_count = 1
    max_repeat = 1
    for i in range(1, len(words)):
        if words[i].lower() == words[i-1].lower():
            consecutive_repeat_count += 1
            max_repeat = max(max_repeat, consecutive_repeat_count)
        else:
            consecutive_repeat_count = 1
            
    return max_repeat > max_consecutive_repeat


def clean_json_markdown(text: str) -> str:
    """
    清洗 markdown json 格式
    例如: ```json { ... } ``` -> { ... }
    """
    text = text.strip()
    # 移除 ```json 或 ```
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    
    # 移除结尾 ```
    if text.endswith("```"):
        text = text[:-3].strip()
    
    return text


def json_format_reward(response: str) -> float:
    """
    检查JSON格式和必需字段奖励 (优化版)
    """
    r = (response or "").strip()
    
    # 1. 提取 JSON 部分 (必须在 think 之后)
    response_clean, has_think = filter_thinking_part(r)
    
    # 2. 清洗 Markdown
    ans_clean = clean_json_markdown(response_clean)
    
    try:
        response_json = json.loads(ans_clean)
        
        if not isinstance(response_json, dict):
            return 0.0

        # 【必要条件1】Answer 字段
        if "answer" not in response_json or not isinstance(response_json.get("answer"), bool):
            return 0.0

        answer = response_json.get("answer")

        # 【必要条件2】Explanation 字段
        if not ("explanation" in response_json and isinstance(response_json.get("explanation"), str)):
            return 0.0
        
        explanation = response_json.get("explanation", "").strip()
        if not explanation:
            return 0.0
            
        if check_format_collapse(explanation):
            return 0.0
            
        if "brief, specific description" in explanation.lower():
            return 0.0

        # 【逻辑分支】根据 Answer 判断
        if answer is False:
            # Answer=False 必须有有效的 edit_prompt
            if "edit_prompt" not in response_json:
                return 0.0
            
            edit_prompt = str(response_json.get("edit_prompt", "")).strip()
            
            if not is_valid_edit_prompt(edit_prompt):
                return 0.0
                
            if check_format_collapse(edit_prompt):
                return 0.0
        
        # Answer=True 时，不需要 edit_prompt，如果存在也不验证
        return 1.0
        
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """计算判断准确性奖励"""
    try:
        # 清洗 markdown
        response_clean = clean_json_markdown(response)
        
        gt_data = json.loads(ground_truth)
        gt_answer = gt_data.get('answer', False)
        
        # 1. 正则优先
        match = re.search(r'"answer"\s*:\s*(true|false)', response_clean, re.IGNORECASE)
        if match:
            extracted_value = match.group(1).lower() == "true"
            return 1.0 if extracted_value == gt_answer else 0.0
            
        # 2. JSON 兜底
        try:
            model_json = json.loads(response_clean)
            if isinstance(model_json, dict) and "answer" in model_json:
                model_val = model_json["answer"]
                if isinstance(model_val, bool):
                    return 1.0 if model_val == gt_answer else 0.0
        except:
            pass
    except:
        pass
    return 0.0


def extract_edit_info(response: str) -> Optional[Dict[str, str]]:
    """
    从响应中提取edit_prompt和explanation (严厉版)
    
    只有在 Thinking 在前、JSON 在后的正确顺序下，才允许提取 Edit Info。
    防止模型跳过思考直接输出 JSON 骗取 Stage 2 奖励。
    """
    try:
        # 1. 严格过滤：必须有 </thinking> 标签
        response_part, has_think_tag = filter_thinking_part(response)
        
        # 如果没有 think 标签，或者标签在最后导致内容为空 -> 视为顺序错误，拒绝提取
        if not response_part.strip() and not has_think_tag:
             # 如果完全没标签，可能是不符合格式，返回 None
             # 注意：如果 filter_thinking_part 返回内容，说明至少找到了标签
             return None

        # 2. 清洗 Markdown
        clean_response = clean_json_markdown(response_part)
        
        # 3. 解析
        try:
            response_json = json.loads(clean_response)
            if isinstance(response_json, dict):
                edit_prompt = str(response_json.get("edit_prompt", "")).strip()
                explanation = str(response_json.get("explanation", "")).strip()
                
                if is_valid_edit_prompt(edit_prompt):
                    return {"edit_prompt": edit_prompt, "explanation": explanation}
        except:
            pass
                
        return None
    except Exception:
        return None


def compute_stage2_reward_api(
    edit_prompt: str,
    original_image_path: str,
    original_prompt: str,
    reward_type: str = "clip",
    edit_config: Optional[dict] = None,
    explanation: Optional[str] = None,
    ground_truth: Optional[str] = None,
) -> dict:
    """通过API计算第二阶段奖励"""
    try:
        edit_client, _, sam3_client = initialize_clients()
        
        if edit_client is None:
            return {"score": 0.0, "success": False, "error": "Image edit client not available"}
        
        edit_prompt_clean = edit_prompt.strip() if edit_prompt else ""
        if not edit_prompt_clean or edit_prompt_clean.lower() in ["remain unchanged", "no edit"]:
            return {"score": 0.0, "success": False, "error": "Invalid edit prompt"}
        
        if not os.path.exists(original_image_path):
            return {"score": 0.0, "success": False, "error": f"Image not found: {original_image_path}"}
            
        original_image = Image.open(original_image_path).convert("RGB")
        
        edit_config = edit_config or {}
        # 默认参数配置
        resolution = edit_config.get("resolution", 1024)
        num_timesteps = edit_config.get("num_timesteps", 40)
        cfg_scale = edit_config.get("cfg_scale", 4.0)
        true_cfg_scale = edit_config.get("true_cfg_scale", 4.0)
        timestep_shift = edit_config.get("timestep_shift", 3.0)
        resolution_scale = edit_config.get("resolution_scale", 0.75)

        # 1. 编辑图像
        edited_image = edit_client.edit_image(
            image=original_image,
            edit_prompt=edit_prompt_clean,
            resolution=resolution,
            num_timesteps=num_timesteps,
            cfg_scale=cfg_scale,
            true_cfg_scale=true_cfg_scale,
            timestep_shift=timestep_shift,
            resolution_scale=resolution_scale,
            model_type="qwen",
        )
        edited_image = edited_image.convert("RGB")
        
        # 2. 计算奖励
        if reward_type == "sam3":
            if sam3_client is None:
                return {"score": 0.0, "success": False, "error": "SAM3 client not available"}
            
            category = "object"
            if ground_truth:
                try:
                    category = json.loads(ground_truth).get("category", "object")
                except:
                    pass
            
            try:
                sam3_res = sam3_client.compute_reward(
                    image=edited_image,
                    prompt=original_prompt,
                    reward_type="sam3",
                    category=category,
                    ground_truth=ground_truth
                )
                return {"score": sam3_res.get("score", 0.0), "success": True, "edited_image": edited_image}
            except Exception as e:
                return {"score": 0.0, "success": False, "error": str(e)}
                
        elif reward_type == "mixed":
            if sam3_client is None:
                return {"score": 0.0, "success": False, "error": "SAM3 client not available"}
            
            base_reward_type = _get_reward_type_for_current_gpu() or os.environ.get("REWARD_TYPE", "omniverifier")
            if base_reward_type in ["mixed", "sam3"]: base_reward_type = "omniverifier"
            
            base_client = get_reward_client_for_type(base_reward_type)
            if base_client is None:
                return {"score": 0.0, "success": False, "error": f"{base_reward_type} client not available"}
            
            category = "object"
            if ground_truth:
                try:
                    category = json.loads(ground_truth).get("category", "object")
                except:
                    pass

            def _run_sam3():
                return sam3_client.compute_reward(
                    image=edited_image, prompt=original_prompt, reward_type="sam3",
                    category=category, ground_truth=ground_truth
                )
            
            def _run_base():
                return base_client.compute_reward(
                    image=edited_image, prompt=original_prompt, reward_type=base_reward_type
                )
            
            with ThreadPoolExecutor(max_workers=2) as exc:
                f1 = exc.submit(_run_sam3)
                f2 = exc.submit(_run_base)
                try:
                    res1 = f1.result()
                    res2 = f2.result()
                    s1 = res1.get("score", 0.0)
                    s2 = res2.get("raw_score", res2.get("score", 0.0))
                    final = 0.5 * s1 + 0.5 * s2
                    return {"score": final, "success": True, "sam3": s1, "base": s2, "edited_image": edited_image}
                except Exception as e:
                    return {"score": 0.0, "success": False, "error": str(e)}
        else:
            target_client = get_reward_client_for_type(reward_type)
            if target_client is None:
                return {"score": 0.0, "success": False, "error": f"{reward_type} client not available"}
            
            try:
                res = target_client.compute_reward(
                    image=edited_image, prompt=original_prompt, reward_type=reward_type
                )
                return {"score": res.get("raw_score", 0.0), "success": True, "edited_image": edited_image}
            except Exception as e:
                return {"score": 0.0, "success": False, "error": str(e)}
                
    except Exception as e:
        return {"score": 0.0, "success": False, "error": str(e)}


def compute_score(
    reward_inputs: list[dict[str, Any]],
    think_format_weight: float = 0.1,
    json_format_weight: float = 0.1,
    stage1_weight: float = 0.4,
    stage2_weight: float = 0.6,
    enable_stage2: bool = True,
    image_dir: Optional[str] = None,
    edit_config: Optional[dict] = None,
    max_workers: Optional[int] = None,
    default_reward_type: Optional[str] = None,
    virtual_correct_reward: float = 0.0,
    **kwargs
) -> list[dict[str, float]]:
    """
    计算两阶段奖励主入口 (API版本)
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch`.")
    
    if default_reward_type is None:
        default_reward_type = os.environ.get("REWARD_TYPE", None)
        
    try:
        edit_client, _, _ = initialize_clients()
    except:
        enable_stage2 = False
        edit_client = None
        
    if max_workers is None:
        max_workers = len(edit_client.endpoints) if edit_client else 1
        
    batch_image_paths = kwargs.get("image_paths", None)
    batch_prompts = kwargs.get("prompts", None)
    
    stage1_results = []
    stage2_tasks = []
    
    # ---------------- Stage 1 Loop ----------------
    for idx, inp in enumerate(reward_inputs):
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", inp.get("response", ""))
        
        # 1. Think Format
        think_score = think_format_reward(response)
        
        # 2. Accuracy
        response_clean, _ = filter_thinking_part(response)
        accuracy_score = accuracy_reward(response_clean, inp["ground_truth"])
        
        # 3. JSON Format
        json_score = json_format_reward(response)
        
        # 【核心修改：门控逻辑】
        # 如果 think_score == 0 (没思考或格式烂)，强制 Accuracy 和 JSON 为 0
        # 并且在下面禁止进入 Stage 2
        if think_score == 0.0:
            accuracy_score = 0.0
            json_score = 0.0
        
        # Stage 1 Reward
        acc_w = max(0.0, 1.0 - think_format_weight)
        s1_rew = acc_w * accuracy_score + think_format_weight * think_score
        
        # 解析 Answer 用于逻辑判断
        model_answer = False
        clean_text = clean_json_markdown(response_clean)
        match = re.search(r'"answer"\s*:\s*(true|false)', clean_text, re.IGNORECASE)
        if match: model_answer = (match.group(1).lower() == "true")
            
        stage1_results.append({
            "idx": idx,
            "think_score": think_score,
            "json_score": json_score,
            "accuracy_score": accuracy_score,
            "stage1_reward": s1_rew,
            "model_answer": model_answer
        })
        
        # 【门控】只有思考了，才允许做后面的事
        allow_stage2 = enable_stage2 and (think_score > 0.0)
        
        if allow_stage2:
            try:
                gt_data = json.loads(inp["ground_truth"])
                gt_answer = gt_data.get("answer", False)
                category = gt_data.get("category", "")
                r_type = default_reward_type or gt_data.get("reward_type") or _CATEGORY_TO_REWARD_TYPE.get(category, "omniverifier")
                
                # 只有 Model=False & GT=False 时计算编辑
                if not model_answer and not gt_answer:
                    # 使用严格版的 extract_edit_info (必须在 think 后面)
                    edit_info = extract_edit_info(response)
                    
                    if edit_info:
                        # 路径解析
                        img_path = ""
                        if "images" in inp and inp["images"]: img_path = inp["images"][0]
                        elif batch_image_paths and idx < len(batch_image_paths): img_path = batch_image_paths[idx]
                        if img_path and image_dir and not os.path.isabs(img_path): img_path = os.path.join(image_dir, img_path)
                        
                        prmt = inp.get("prompt", "") or gt_data.get("prompt", "")
                        if not prmt and batch_prompts: prmt = batch_prompts[idx]
                        
                        if img_path and os.path.exists(img_path):
                            stage2_tasks.append({
                                "idx": idx,
                                "type": "compute",
                                "edit_prompt": edit_info["edit_prompt"],
                                "explanation": edit_info["explanation"],
                                "image_path": img_path,
                                "prompt": prmt,
                                "reward_type": r_type,
                                "edit_config": edit_config,
                                "json_score": json_score,
                                "ground_truth": inp["ground_truth"],
                                "is_false_false": True
                            })
                        else:
                            stage2_tasks.append({"idx": idx, "type": "error", "msg": "Image missing", "json_score": json_score})
                    else:
                        # 无法提取指令 (或顺序错误被 extract_edit_info 拦截)
                        stage2_tasks.append({"idx": idx, "type": "error", "msg": "No valid edit info or Wrong Order", "json_score": json_score})
                        
                elif model_answer and gt_answer:
                    # TP
                    s2_val = (1.0 - json_format_weight) * virtual_correct_reward + json_format_weight * json_score
                    stage2_tasks.append({"idx": idx, "type": "TP", "val": s2_val, "json_score": json_score})
                
                elif model_answer and not gt_answer:
                    # FP
                    stage2_tasks.append({"idx": idx, "type": "FP", "val": 0.0, "json_score": 0.0})
                    
                elif not model_answer and gt_answer:
                    # FN
                    stage2_tasks.append({"idx": idx, "type": "FN", "val": 0.0, "json_score": 0.0})
                    
            except Exception as e:
                stage2_tasks.append({"idx": idx, "type": "error", "msg": str(e), "json_score": 0.0})
        elif enable_stage2:
            # 本来开了 stage2 但因为没思考被拦住了
            stage2_tasks.append({
                "idx": idx,
                "type": "error",
                "msg": "Skipped: No thinking",
                "json_score": 0.0 
            })

    # ---------------- Stage 2 Execution ----------------
    stage2_results_map = {}
    compute_tasks = [t for t in stage2_tasks if t["type"] == "compute"]
    direct_tasks = [t for t in stage2_tasks if t["type"] != "compute"]
    
    for t in direct_tasks:
        idx = t["idx"]
        if t["type"] in ["TP", "FP", "FN"]:
            stage2_results_map[idx] = {"reward": t["val"], "details": {"type": t["type"], "json_score": t["json_score"]}}
        else:
            # error case
            r_val = json_format_weight * t.get("json_score", 0.0)
            stage2_results_map[idx] = {"reward": r_val, "details": {"type": "error", "reason": t.get("msg"), "json_score": t.get("json_score", 0.0)}}

    if compute_tasks and edit_client:
        def _process(task):
            try:
                res = compute_stage2_reward_api(
                    edit_prompt=task["edit_prompt"],
                    original_image_path=task["image_path"],
                    original_prompt=task["prompt"],
                    reward_type=task["reward_type"],
                    edit_config=task["edit_config"],
                    explanation=task["explanation"],
                    ground_truth=task["ground_truth"]
                )
                if res["success"]:
                    edit_sc = res["score"]
                    w_ed = max(0.0, 1.0 - json_format_weight)
                    fin = json_format_weight * task["json_score"] + w_ed * edit_sc
                    return task["idx"], {"reward": fin, "details": {"type": "TN", "edit_reward": edit_sc, "json_score": task["json_score"], "is_false_false": True, "success": True}}
                else:
                    return task["idx"], {"reward": json_format_weight * task["json_score"], "details": {"type": "error", "reason": res.get("error"), "json_score": task["json_score"]}}
            except Exception as e:
                return task["idx"], {"reward": 0.0, "details": {"type": "error", "reason": str(e)}}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process, t) for t in compute_tasks]
            for f in as_completed(futures):
                try:
                    idx, res = f.result()
                    stage2_results_map[idx] = res
                except:
                    pass

    # ---------------- Final Merge & Stats ----------------
    final_scores = []
    stats = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "Error": 0, "Reasons": Counter()}
    
    for idx, s1_res in enumerate(stage1_results):
        s2_res = stage2_results_map.get(idx, {"reward": 0.0, "details": {"type": "skipped"}})
        overall = stage1_weight * s1_res["stage1_reward"] + stage2_weight * s2_res["reward"]
        
        # Stats accumulation
        dt = s2_res["details"].get("type", "unknown")
        if dt in stats: 
            stats[dt] += 1
        elif dt == "error": 
            stats["Error"] += 1
            stats["Reasons"][s2_res["details"].get("reason", "unk")] += 1
            
        final_scores.append({
            "overall": overall,
            "stage1_reward": s1_res["stage1_reward"],
            "stage2_reward": s2_res["reward"],
            "think_format": s1_res["think_score"],
            "json_format": s1_res["json_score"],
            "stage1_accuracy": s1_res["accuracy_score"],
            "stage2_accuracy": s2_res["details"].get("edit_reward", 0.0),
            "TN_counts": 1.0 if s2_res["details"].get("is_false_false") else 0.0,
            "stage2_details": s2_res["details"]
        })
    
    # 打印详细日志
    total = len(final_scores)
    print(f"\n[Reward Stats] Batch: {total}")
    for k in ["TP", "TN", "FP", "FN", "Error"]:
        if stats[k] > 0:
            print(f"  {k}: {stats[k]} ({stats[k]/total*100:.1f}%)")
            if k == "Error":
                for r, c in stats["Reasons"].most_common(3):
                    print(f"    - {r}: {c}")
    print("-" * 40)
    
    return final_scores