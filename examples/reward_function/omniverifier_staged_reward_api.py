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

# 从 examples/reward_function/ 向上回退到 project root/，然后进入 distributed_services/
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
    """
    安全地从环境变量获取整数值，自动去除引号
    
    Args:
        key: 环境变量键
        default: 默认值
    
    Returns:
        整数值
    """
    value = os.environ.get(key, str(default))
    # 去除可能的引号（配置文件解析可能遗留）
    value = value.strip().strip('"').strip("'")
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _get_env_bool(key: str, default: bool = True) -> bool:
    """
    安全地从环境变量获取布尔值，自动去除引号
    
    Args:
        key: 环境变量键
        default: 默认值
    
    Returns:
        布尔值
    """
    value = os.environ.get(key, str(default).lower())
    # 去除可能的引号
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
    """
    根据 REWARD_TYPE_PER_GPU 环境变量和当前 GPU ID 获取奖励类型
    
    Returns:
        当前 GPU 对应的奖励类型，如果无法确定则返回 None
    """
    reward_type_per_gpu = os.environ.get("REWARD_TYPE_PER_GPU", "")
    if not reward_type_per_gpu:
        return None
    
    # 尝试获取当前 GPU ID
    # 优先级：LOCAL_RANK > CUDA_VISIBLE_DEVICES 的第一个设备 > 0（默认）
    gpu_id = 0
    local_rank = os.environ.get("LOCAL_RANK", None)
    if local_rank is not None:
        try:
            gpu_id = int(local_rank)
        except (ValueError, TypeError):
            gpu_id = 0
    else:
        # 如果没有 LOCAL_RANK，尝试从 CUDA_VISIBLE_DEVICES 推断
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            # CUDA_VISIBLE_DEVICES 通常是逗号分隔的设备列表
            # 如果设置了，说明当前进程只能看到这些设备，第一个设备对应 gpu_id=0
            # 但实际物理 GPU ID 可能不同，这里假设 LOCAL_RANK 更准确
            pass
    
    # 解析 REWARD_TYPE_PER_GPU 格式："0:omniverifier,1:omniverifier,2:omniverifier,3:sam3"
    try:
        mappings = {}
        for mapping in reward_type_per_gpu.split(","):
            mapping = mapping.strip()
            if ":" in mapping:
                gpu_str, reward_type = mapping.split(":", 1)
                gpu_str = gpu_str.strip()
                reward_type = reward_type.strip()
                try:
                    gpu_num = int(gpu_str)
                    mappings[gpu_num] = reward_type
                except (ValueError, TypeError):
                    continue
        
        # 返回当前 GPU ID 对应的奖励类型
        return mappings.get(gpu_id)
    except Exception as e:
        # 解析失败，返回 None
        return None


def initialize_clients():
    """初始化API客户端（从环境变量，支持配置文件热重载）"""
    global _edit_client, _reward_client, _sam3_reward_client, _client_warning_printed
    
    if not API_CLIENT_AVAILABLE:
        raise RuntimeError("API client not available. Cannot initialize clients.")
    
    # 如果客户端已存在，检查配置文件是否需要重新加载
    if _edit_client is not None or _reward_client is not None or _sam3_reward_client is not None:
        try:
            from clients.api_client import _check_and_reload_config
            if _check_and_reload_config(_edit_client, _reward_client, _sam3_reward_client):
                print("[INFO] 检测到配置文件修改，已自动更新客户端端点")
        except Exception as e:
            pass
    
    if _edit_client is None or _reward_client is None or _sam3_reward_client is None:
        _edit_client, _reward_client, _sam3_reward_client = create_clients_from_env()
    
    # 优化：只在第一次检测到客户端缺失时打印警告，避免重复打印
    if _edit_client is None and not _client_warning_printed.get("edit_client", False):
        print("Warning: Image edit client not initialized. Check EDIT_SERVER_ENDPOINTS environment variable.")
        _client_warning_printed["edit_client"] = True
    
    if _reward_client is None and not _client_warning_printed.get("reward_client", False):
        print("Warning: Reward client not initialized. Check REWARD_SERVER_ENDPOINTS environment variable.")
        _client_warning_printed["reward_client"] = True
        
    # if _sam3_reward_client is None:
    #     print("Warning: SAM3 reward client not initialized. Check SAM3_REWARD_SERVER_ENDPOINTS environment variable.")
    
    return _edit_client, _reward_client, _sam3_reward_client


def get_reward_client_for_type(reward_type: str):
    """
    根据奖励类型获取正确的奖励客户端
    
    Args:
        reward_type: 奖励类型（clip、omniverifier、sam3）
    
    Returns:
        对应的奖励客户端，如果不存在则返回 None
    """
    global _reward_client, _sam3_reward_client, _client_warning_printed
    
    if reward_type == "sam3":
        # 复用 RewardClient，但使用 compute_sam3_reward 接口
        return _sam3_reward_client
    elif reward_type == "omniverifier":
        # 尝试从 OMNIVERIFIER_REWARD_SERVER_ENDPOINTS 创建 OmniVerifier 客户端
        omniverifier_endpoints = os.environ.get("OMNIVERIFIER_REWARD_SERVER_ENDPOINTS", "")
        if omniverifier_endpoints:
            try:
                from clients.api_client import RewardClient
                endpoints = [e.strip() for e in omniverifier_endpoints.split(",") if e.strip()]
                if endpoints:
                    timeout = _get_env_int("API_REQUEST_TIMEOUT", 180)
                    max_retries = _get_env_int("API_MAX_RETRIES", 3)
                    health_check_timeout = _get_env_int("API_HEALTH_CHECK_TIMEOUT", 5)
                    health_check_interval = _get_env_int("API_HEALTH_CHECK_INTERVAL", 600)
                    enable_health_check = _get_env_bool("API_ENABLE_HEALTH_CHECK", True)
                    return RewardClient(
                        endpoints,
                        timeout=timeout,
                        max_retries=max_retries,
                        health_check_timeout=health_check_timeout,
                        health_check_interval=health_check_interval,
                        enable_health_check=enable_health_check
                    )
            except Exception as e:
                # 优化：只在第一次打印，避免重复打印
                if not _client_warning_printed.get("omniverifier_client_error", False):
                    print(f"[WARNING] 无法创建 OmniVerifier 客户端: {e}")
                    _client_warning_printed["omniverifier_client_error"] = True
        
        # 如果没有单独的 OmniVerifier 端点，检查环境变量 REWARD_TYPE
        # 如果 REWARD_TYPE 是 "mixed" 或 "clip"，说明实际使用的是 CLIP 端点
        # 这种情况下应该返回 None 或给出明确的错误提示，而不是回退到可能不支持 omniverifier 的客户端
        env_reward_type = os.environ.get("REWARD_TYPE", "").lower()
        if env_reward_type in ["mixed", "clip"]:
            # 优化：只在第一次打印，避免重复打印
            if not _client_warning_printed.get("omniverifier_endpoints_empty", False):
                print(f"[WARNING] 请求 omniverifier 奖励类型，但 OMNIVERIFIER_REWARD_SERVER_ENDPOINTS 为空")
                print(f"[WARNING] 当前 REWARD_TYPE={env_reward_type}，实际可用的端点可能只支持 clip 类型")
                print(f"[WARNING] 建议：1) 部署 OmniVerifier 奖励服务并配置 OMNIVERIFIER_REWARD_SERVER_ENDPOINTS")
                print(f"[WARNING]         2) 或者将 default_reward_type 设置为 'clip'")
                _client_warning_printed["omniverifier_endpoints_empty"] = True
            # 返回 None，让调用方处理错误
            return None
        
        # 如果环境变量 REWARD_TYPE 是 "omniverifier"，尝试使用默认的 reward_client
        # 但需要确保它支持 omniverifier 类型（这种情况下可能配置有问题）
        if env_reward_type == "omniverifier":
            # 优化：只在第一次打印，避免重复打印
            if not _client_warning_printed.get("reward_type_fallback", False):
                print(f"[WARNING] REWARD_TYPE=omniverifier 但 OMNIVERIFIER_REWARD_SERVER_ENDPOINTS 为空，回退到 REWARD_SERVER_ENDPOINTS")
                print(f"[WARNING] 请确保 REWARD_SERVER_ENDPOINTS 中的端点支持 omniverifier 类型")
                _client_warning_printed["reward_type_fallback"] = True
        
        return _reward_client
    else:
        # 默认 CLIP，使用默认的 reward_client
        # 但如果 REWARD_SERVER_ENDPOINTS 中包含 CLIP 端点，应该使用 CLIP_REWARD_SERVER_ENDPOINTS
        clip_endpoints = os.environ.get("CLIP_REWARD_SERVER_ENDPOINTS", "")
        if clip_endpoints:
            try:
                from clients.api_client import RewardClient
                endpoints = [e.strip() for e in clip_endpoints.split(",") if e.strip()]
                if endpoints:
                    timeout = _get_env_int("API_REQUEST_TIMEOUT", 180)
                    max_retries = _get_env_int("API_MAX_RETRIES", 3)
                    health_check_timeout = _get_env_int("API_HEALTH_CHECK_TIMEOUT", 5)
                    health_check_interval = _get_env_int("API_HEALTH_CHECK_INTERVAL", 600)
                    enable_health_check = _get_env_bool("API_ENABLE_HEALTH_CHECK", True)
                    return RewardClient(
                        endpoints,
                        timeout=timeout,
                        max_retries=max_retries,
                        health_check_timeout=health_check_timeout,
                        health_check_interval=health_check_interval,
                        enable_health_check=enable_health_check
                    )
            except Exception as e:
                # 优化：只在第一次打印，避免重复打印
                if not _client_warning_printed.get("clip_client_error", False):
                    print(f"[WARNING] 无法创建 CLIP 客户端: {e}")
                    _client_warning_printed["clip_client_error"] = True
        return _reward_client


def filter_thinking_part(response, eos_token=None):
    """
    提取answer部分（去除think标签）
    
    支持两种格式：
    1. 标准格式：<think>think内容</think>answer
    2. 只有结束标签：think内容</think>answer（模型偏好格式）
    """
    response_start = 0
    success = False
    think_tag_start = '<think>'
    think_tag_end = '</think>'
    
    # 查找结束标签（从后往前找最后一个）
    think_end = response.rfind(think_tag_end)
    
    if think_end != -1:
        # 找到了结束标签，提取结束标签之后的内容
        response_start = think_end + len(think_tag_end)
        success = True
        
        # 可选：检查是否有开始标签（用于兼容标准格式）
        think_start = response.find(think_tag_start, 0, think_end)
        # 如果有开始标签且位置正确，说明是标准格式；否则是只有结束标签的格式
        # 两种格式都支持，所以这里不需要额外处理
    else:
        # 没有找到结束标签，尝试查找开始标签（向后兼容）
        think_start = response.find(think_tag_start)
        if think_start != -1:
            # 有开始标签但没有结束标签，尝试从开始标签之后提取
            response_start = think_start + len(think_tag_start)
            success = True
    
    if eos_token is not None:
        response_end = response.find(eos_token, response_start)
    else:
        response_end = len(response)
    response = response[response_start:response_end]
    return response, success


def think_format_reward(response: str) -> float:
    """
    检查think标签格式奖励
    
    支持两种格式：
    1. 标准格式：<think>think内容</think>answer
    2. 只有结束标签：think内容</think>answer（模型偏好格式）
    
    只要检测到结束标签 `</think>` 就认为有 think 格式
    """
    r = (response or "").strip()
    
    think_tag_start = "<think>"
    think_tag_end = "</think>"
    
    # 查找结束标签（从后往前找最后一个）
    think_end = r.rfind(think_tag_end)
    
    if think_end == -1:
        # 没有找到结束标签，返回0分
        return 0.0
    
    # 找到了结束标签，提取think内容和answer部分
    # 查找开始标签（在结束标签之前）
    think_start = r.find(think_tag_start, 0, think_end)
    
    if think_start != -1:
        # 标准格式：有开始标签和结束标签
        think = r[think_start + len(think_tag_start):think_end]
    else:
        # 只有结束标签的格式：从开头到结束标签之前都是think内容
        think = r[:think_end]
    
    # 提取结束标签之后的内容作为answer
    ans = r[think_end + len(think_tag_end):].strip()
    
    # 检查：think内容非空，answer非空，且answer中不包含think开始标签（避免嵌套）
    if think.strip() and ans.strip() and think_tag_start not in ans:
        return 1.0
    else:
        return 0.0


def is_valid_edit_prompt(edit_prompt: str) -> bool:
    """
    检查edit_prompt是否有效（统一的验证逻辑，供json_format_reward和extract_edit_info使用）
    
    Args:
        edit_prompt: 待检查的edit_prompt字符串
    
    Returns:
        True: edit_prompt有效，可以用于编辑
        False: edit_prompt无效，不应该用于编辑
    """
    if not edit_prompt:
        return False
    
    edit_prompt_clean = edit_prompt.strip()
    
    # 1. 检查是否为空（包括只有空格的情况）
    if not edit_prompt_clean:
        return False
    
    # 2. 检查长度（确保是有效的编辑指令，而不是无意义的短文本）
    if len(edit_prompt_clean) <= 10:
        return False
    
    # 3. 检查是否是无效值
    edit_prompt_lower = edit_prompt_clean.lower()
    if edit_prompt_lower in ["remain unchanged", "no edit", ""]:
        return False
    
    # 4. 检查是否是模板文本（防止模型复制模板描述）
    template_patterns = [
        "a concrete, location-specific editing instruction",
        "concrete, location-specific editing instruction to fix the error",
        "provide a concrete, location-specific editing instruction",
        "location-specific editing instruction",
    ]
    for pattern in template_patterns:
        if pattern in edit_prompt_lower:
            return False  # 检测到模板文本
    
    # 5. 检查是否包含实际的编辑动作词（放宽检查，只要包含动作词即可，给模型更多探索空间）
    action_words = ['add', 'remove', 'replace', 'change', 'move', 'delete', 'place', 'position', 'shift', 'make', 'modify', 'update']
    has_action_word = any(word in edit_prompt_lower for word in action_words)
    
    # 如果没有检测到动作词，可能是无效的edit_prompt
    # 但允许一些其他可能的编辑指令格式（如 "set", "adjust" 等），只要不是模板文本即可
    if not has_action_word:
        # 如果长度足够且不是模板文本，也允许通过（给模型探索空间）
        # 但至少要确保不是明显的无效文本
        if len(edit_prompt_clean) < 20:
            return False  # 太短且没有动作词，可能是无效的
    
    return True


def check_format_collapse(text: str, min_words: int = 5, max_consecutive_repeat: int = 5) -> bool:
    """
    检测文本格式崩溃：如果某个单词连续重复超过阈值，说明格式崩溃
    
    Args:
        text: 要检测的文本
        min_words: 最小单词数，少于这个数量不检测
        max_consecutive_repeat: 最大允许的连续重复次数，超过则判定为格式崩溃
    
    Returns:
        True: 格式崩溃
        False: 格式正常
    """
    if not text:
        return False
    
    # 【修复1】检测字符重复（如空格、引号等字符的连续重复）
    # 这是导致模型输出 " " " " " " ... 塌缩的主要原因
    if len(text) > 10:
        # 检测连续重复的字符（空格、引号等）
        consecutive_char_count = 1
        max_char_repeat = 1
        for i in range(1, len(text)):
            if text[i] == text[i-1] and text[i] in [' ', '"', "'", '\n', '\t']:
                consecutive_char_count += 1
                max_char_repeat = max(max_char_repeat, consecutive_char_count)
            else:
                consecutive_char_count = 1
        
        # 如果连续重复字符超过阈值（如连续20个空格），判定为格式崩溃
        if max_char_repeat > 20:
            return True
    
    # 【修复2】检测单词重复（原有逻辑）
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


def json_format_reward(response: str, require_edit_prompt_for_true: bool = False) -> float:
    """
    检查JSON格式和必需字段奖励
    
    Args:
        response: 模型响应字符串
        require_edit_prompt_for_true: 是否要求answer=true时也必须有edit_prompt字段
                                      - False: 当前设计，true只需要answer和explanation
                                      - True: 统一要求，无论true/false都需要三个字段（降低学习难度）
    
    Returns:
        0.0-1.0: JSON格式奖励分数
    """
    r = (response or "").strip()
    
    think_tag_start = "<think>"
    think_tag_end = "</think>"
    
    # 查找结束标签（从后往前找最后一个）
    think_end = r.rfind(think_tag_end)
    
    if think_end != -1:
        # 找到了结束标签，提取结束标签之后的内容作为JSON
        ans = r[think_end + len(think_tag_end):].strip()
    else:
        # 没有找到结束标签，尝试查找开始标签（向后兼容）
        think_start = r.find(think_tag_start)
        if think_start != -1:
            # 有开始标签但没有结束标签，从开始标签之后提取
            ans = r[think_start + len(think_tag_start):].strip()
        else:
            # 没有think标签，直接使用整个响应
            ans = r
    
    try:
        response_json = json.loads(ans.strip())
        
        # 检查 response_json 是否是字典类型（防止双重编码导致返回字符串）
        if not isinstance(response_json, dict):
            return 0.0

        # 【第一个必要条件】answer字段必须存在且为bool类型，否则直接返回0.0
        # 这是获得JSON格式奖励的第一个前提条件，后续所有检查都基于此
        if "answer" not in response_json or not isinstance(response_json.get("answer"), bool):
            return 0.0

        # 获取answer，用于后续判断
        answer = response_json.get("answer")

        # NOTE:
        # TP情况不计算json格式奖励了，取消释掉return 0.0即可
        # if answer is True:
        #     return 0.0

        # 检查explanation字段必须存在
        if not ("explanation" in response_json and isinstance(response_json.get("explanation"), str)):
            return 0.0
        
        # 获取explanation字段，用于后续判断
        explanation = response_json.get("explanation", "").strip()
        

        # 检测explanation字段的格式崩溃（如果非空）
        if explanation and check_format_collapse(explanation):
            return 0.0  # explanation格式崩溃，返回 0 分

        # 如果answer=false，explanation必须非空
        if answer is False:
            if not explanation:
                return 0.0
        
        # 检测explanation字段的模板文本（防止模式塌缩）- 无论 answer=true 还是 false，只要 explanation 非空就检测
        if explanation:
            explanation_lower = explanation.lower().strip()
            
            # 检测模板文本模式
            # 模板文本："A brief, specific description of the main error (if answer is false)."
            # 1. 检测完整的模板句子（精确匹配）
            explicit_template_patterns = [
                "a brief, specific description of the main error (if answer is false)",
                "a brief, specific description of the main error",
                "brief, specific description of the main error (if answer is false)",
                "brief, specific description of the main error",
            ]
            for pattern in explicit_template_patterns:
                # 精确匹配：整个explanation就是模板文本，或者以模板文本开头/结尾
                if explanation_lower == pattern or explanation_lower.startswith(pattern + ".") or explanation_lower.startswith(pattern + ","):
                    return 0.0  # 检测到明确的模板文本，返回 0 分
            
            # 2. 检测模板的关键片段组合（仅在文本很短时判定，避免误判正常文本）
            has_brief_specific = "brief, specific description" in explanation_lower or "brief specific description" in explanation_lower
            has_main_error = "of the main error" in explanation_lower
            has_if_answer_false = "(if answer is false)" in explanation_lower or "if answer is false" in explanation_lower
            
            # 如果文本很短（<80字符）且同时包含多个模板片段，判定为模板
            # 这样可以避免将正常的长文本误判为模板
            if len(explanation) < 80 and (has_brief_specific and has_main_error):
                return 0.0
            
            # 3. 如果包含完整的模板描述（包含所有关键片段），且文本较短，判定为模板
            if len(explanation) < 120 and has_brief_specific and has_main_error and has_if_answer_false:
                return 0.0
                
        # 检查edit_prompt字段
        needs_edit_prompt = (answer is False) or require_edit_prompt_for_true
        
        if needs_edit_prompt:
            # 检查edit_prompt字段是否存在
            if not ("edit_prompt" in response_json and isinstance(response_json.get("edit_prompt"), str)):
                return 0.0
            
            edit_prompt = response_json.get("edit_prompt", "").strip()
            
            # 如果answer=false，edit_prompt必须非空且长度大于10（确保是有效的编辑指令）
            if answer is False:
                if not edit_prompt or len(edit_prompt) <= 10:
                    return 0.0
            
            # 检测edit_prompt字段的格式崩溃（如果非空）
            if edit_prompt and check_format_collapse(edit_prompt):
                return 0.0  # edit_prompt格式崩溃，返回 0 分 

            # 检测edit_prompt的有效性（使用与extract_edit_info相同的检查逻辑，确保一致性）
            # 如果answer=false，edit_prompt必须有效才能获得json格式奖励
            if answer is False:
                if not is_valid_edit_prompt(edit_prompt):
                    return 0.0  # edit_prompt无效，返回 0 分
            # 如果answer=true，edit_prompt如果存在也应该有效（可选检查）
            elif edit_prompt and not is_valid_edit_prompt(edit_prompt):
                # answer=true时，edit_prompt允许为空，但如果提供了非空内容，也应该有效
                return 0.0  # edit_prompt无效，返回 0 分

        return 1.0
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """计算判断准确性奖励"""
    try:
        answer = json.loads(ground_truth)
        match = re.search(r'"answer"\s*:\s*(true|false)', response, re.IGNORECASE)
        if match and response is not None:
            extracted_value = match.group(1).lower() == "true"
            is_same = extracted_value == answer.get('answer', False)
            if is_same:
                return 1.0
            else:
                return 0.0
    except:
        pass
    return 0.0


def extract_edit_info(response: str) -> Optional[Dict[str, str]]:
    """
    从响应中提取edit_prompt和explanation
    
    使用与json_format_reward相同的验证逻辑（is_valid_edit_prompt），确保一致性
    
    更鲁棒的提取策略：
    1. 首先尝试从 `</think>` 之后提取JSON
    2. 如果失败，尝试从整个响应中查找JSON（可能JSON在think内容中）
    3. 如果还是失败，尝试从think内容中提取JSON（模型可能在think中写了JSON）
    """
    try:
        # 策略1：尝试从 `</think>` 之后提取JSON（标准位置）
        response_clean, has_think_tag = filter_thinking_part(response)
        
        # 尝试解析JSON
        try:
            response_json = json.loads(response_clean.strip())
            if isinstance(response_json, dict):
                edit_prompt = response_json.get("edit_prompt", "")
                explanation = response_json.get("explanation", "")
        
                if is_valid_edit_prompt(edit_prompt):
                    return {
                        "edit_prompt": edit_prompt.strip(),
                        "explanation": explanation,
                    }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        
        # 策略2：如果策略1失败，尝试从整个响应中查找JSON块
        # 查找JSON的开始和结束位置（使用大括号匹配）
        json_start = response.find('{')
        if json_start != -1:
            # 从第一个 { 开始，尝试找到匹配的 }
            brace_count = 0
            json_end = -1
            for i in range(json_start, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end != -1:
                try:
                    json_str = response[json_start:json_end]
                    response_json = json.loads(json_str)
                    if isinstance(response_json, dict):
                        edit_prompt = response_json.get("edit_prompt", "")
                        explanation = response_json.get("explanation", "")
                        
                        if is_valid_edit_prompt(edit_prompt):
                            return {
                                "edit_prompt": edit_prompt.strip(),
                                "explanation": explanation,
                            }
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
        
        # 策略3：如果策略2也失败，尝试从think内容中提取JSON
        # 查找think内容（在 `</think>` 之前）
        think_tag_end = '</think>'
        think_end = response.rfind(think_tag_end)
        if think_end != -1:
            think_content = response[:think_end]
            # 在think内容中查找JSON
            json_start = think_content.find('{')
            if json_start != -1:
                brace_count = 0
                json_end = -1
                for i in range(json_start, len(think_content)):
                    if think_content[i] == '{':
                        brace_count += 1
                    elif think_content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end != -1:
                    try:
                        json_str = think_content[json_start:json_end]
                        response_json = json.loads(json_str)
                        if isinstance(response_json, dict):
                            edit_prompt = response_json.get("edit_prompt", "")
                            explanation = response_json.get("explanation", "")
                            
                            if is_valid_edit_prompt(edit_prompt):
                                return {
                                    "edit_prompt": edit_prompt.strip(),
                                    "explanation": explanation,
                                }
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass
        
        # 所有策略都失败，返回None
        return None
        
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError, KeyError, Exception):
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
    """
    通过API计算第二阶段奖励
    
    Args:
        edit_prompt: 编辑指令
        original_image_path: 原始图像路径
        original_prompt: 原始prompt
        reward_type: 奖励类型
        edit_config: 编辑配置（resolution, num_timesteps等）
        explanation: 问题解释（可选，如果提供会构造与bagel_editor.py一致的edit_instruction）
    
    Returns:
        {
            "score": 奖励分数,
            "success": 是否成功,
            "error": 错误信息（如果有）
        }
    """
    try:
        # 初始化客户端
        edit_client, reward_client, sam3_client = initialize_clients()
        
        if edit_client is None:
            error_msg = "Image edit client not available"
            print(f"[ERROR] compute_stage2_reward_api: {error_msg}", flush=True)
            return {"score": 0.0, "success": False, "error": error_msg}
        
        # 安全检查：确保客户端有必要的属性（防止热重载导致的版本不一致）
        if not hasattr(edit_client, 'health_check_timeout'):
            error_msg = f"ImageEditClient missing required attributes. Please restart the training process."
            print(f"[ERROR] compute_stage2_reward_api: {error_msg}", flush=True)
            return {"score": 0.0, "success": False, "error": error_msg}

        # 注意：reward_client 只在 clip/omniverifier 模式下需要
        # 在 sam3 和 mixed 模式下，奖励客户端会在函数内部根据 reward_type 自动创建
        # 所以这里不检查 reward_client，而是在具体使用时检查
        
        # 防御性检查：如果edit_prompt为空，直接返回0分（不应该发生，但作为安全措施）
        edit_prompt_clean = edit_prompt.strip() if edit_prompt else ""
        if not edit_prompt_clean or edit_prompt_clean.lower() in ["remain unchanged", "no edit"]:
            return {
                "score": 0.0,
                "success": False,
                "error": "edit_prompt is empty or invalid"
            }
        
        # 1. 加载原始图像
        if not os.path.exists(original_image_path):
            return {
                "score": 0.0,
                "success": False,
                "error": f"Image file not found: {original_image_path}"
            }
        if not os.path.isfile(original_image_path):
            return {
                "score": 0.0,
                "success": False,
                "error": f"Path is not a file: {original_image_path}"
            }
        original_image = Image.open(original_image_path).convert("RGB")
        
        edit_instruction = edit_prompt_clean
        
        # 3. 调用图像编辑服务
        edit_config = edit_config or {}
        resolution = edit_config.get("resolution", 1024)
        num_timesteps = edit_config.get("num_timesteps", 50)
        cfg_scale = edit_config.get("cfg_scale", 4.0)
        true_cfg_scale = edit_config.get("true_cfg_scale", 4.0)
        timestep_shift = edit_config.get("timestep_shift", 3.0)
        resolution_scale = edit_config.get("resolution_scale", 0.50)

        edited_image = edit_client.edit_image(
            image=original_image,
            edit_prompt=edit_instruction,  # 使用构造好的edit_instruction
            resolution=resolution,
            num_timesteps=num_timesteps,
            cfg_scale=cfg_scale,
            true_cfg_scale=true_cfg_scale,
            timestep_shift=timestep_shift,
            resolution_scale=resolution_scale,
            model_type="qwen",
        )
        
        # 确保图片格式为RGB（与CLIP/SAM3/OmniVerifier服务器端处理一致）
        edited_image = edited_image.convert("RGB")
        
        # 3. 调用奖励计算服务（按 reward_type 选择客户端）
        score = 0.0
        if reward_type == "sam3":
            if sam3_client is None:
                return {"score": 0.0, "success": False, "error": "SAM3 reward client not available"}
            
            category = None
            if ground_truth:
                try:
                    gt_data = json.loads(ground_truth)
                    if isinstance(gt_data, dict):
                        category = gt_data.get("category", "")
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass
            else:
                print("No ground truth provided, using default category: object", flush=True)
                pass

            if not category:
                print("No category found in ground truth, using default category: object", flush=True)
                category = "object"
            
            try:
                sam3_result = sam3_client.compute_reward(
                    image=edited_image,
                    prompt=original_prompt,
                    reward_type="sam3", 
                    category=category,
                    ground_truth=ground_truth,
                )
                # compute_reward 成功时返回 {"score": ..., "raw_score": ..., "reward_type": ...}
                # 失败时会抛出 RuntimeError
                score = sam3_result.get("score", 0.0)
            except Exception as e:
                # RewardClient.compute_reward 如果失败会抛出 RuntimeError
                return {"score": 0.0, "success": False, "error": str(e)}
        elif reward_type == "mixed":
            # 并行计算 sam3 与 clip/omniverifier，默认各占 50%
            if sam3_client is None:
                error_msg = "SAM3 reward client not available"
                print(f"[ERROR] compute_stage2_reward_api: {error_msg}", flush=True)
                return {"score": 0.0, "success": False, "error": error_msg}
            
            category = None
            if ground_truth:
                try:
                    gt_data = json.loads(ground_truth)
                    if isinstance(gt_data, dict):
                        category = gt_data.get("category", "")
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass
            else:
                # 优化：减少日志输出，只在必要时打印
                pass

            if not category:
                category = "object"

            # 根据 REWARD_TYPE_PER_GPU 或默认类型选择基础奖励类型
            # 注意：当 reward_type == "mixed" 时，base_reward_type 必须是 "clip" 或 "omniverifier"
            # 不能是 "mixed"，因为 CLIP/OmniVerifier 服务不支持 "mixed" 类型
            
            # 优先从 REWARD_TYPE_PER_GPU 获取当前 GPU 对应的奖励类型
            base_reward_type = _get_reward_type_for_current_gpu()
            
            # 如果 REWARD_TYPE_PER_GPU 没有配置或解析失败，从 REWARD_TYPE 获取
            if base_reward_type is None:
                base_reward_type = os.environ.get("REWARD_TYPE", "omniverifier")
            if base_reward_type == "mixed":
                    base_reward_type = "omniverifier"  # mixed 模式下默认使用 omniverifier
            
            # 确保 base_reward_type 不是 "mixed" 或 "sam3"（sam3 已经在 _call_sam3 中处理）
            if base_reward_type in ["mixed", "sam3"]:
                base_reward_type = "omniverifier"  # 回退到 omniverifier
            
            # 获取正确的基础奖励客户端
            base_reward_client = get_reward_client_for_type(base_reward_type)
            if base_reward_client is None:
                error_msg = f"{base_reward_type} reward client not available"
                print(f"[ERROR] compute_stage2_reward_api: {error_msg}", flush=True)
                return {"score": 0.0, "success": False, "error": error_msg}

            def _call_sam3():
                try:
                    return sam3_client.compute_reward(
                        image=edited_image,
                        prompt=original_prompt,
                        reward_type="sam3",
                        category=category,
                        ground_truth=ground_truth,
                    )
                except RuntimeError as e:
                    error_msg = str(e) if e and str(e) else "Unknown RuntimeError in _call_sam3"
                    print(f"[ERROR] _call_sam3 RuntimeError: {error_msg}", flush=True)
                    return {"success": False, "error": error_msg, "score": 0.0}
                except Exception as e:
                    # 捕获其他类型的异常（如图片解码错误等）
                    import traceback
                    error_msg = str(e) if e and str(e) else "Unknown exception in _call_sam3"
                    error_detail = f"{error_msg}\n{traceback.format_exc()}"
                    print(f"[ERROR] _call_sam3 异常: {error_detail}", flush=True)
                    return {"success": False, "error": error_msg, "score": 0.0}

            def _call_base():
                try:
                    result = base_reward_client.compute_reward(
                        image=edited_image,
                        prompt=original_prompt,
                        reward_type=base_reward_type,
                    )
                    return result
                except RuntimeError as e:
                    # RewardClient.compute_reward如果失败会抛出RuntimeError
                    # 捕获异常并返回错误信息
                    error_msg = str(e) if e and str(e) else "Unknown RuntimeError in _call_base"
                    print(f"[ERROR] _call_base RuntimeError: {error_msg}", flush=True)
                    return {"success": False, "error": error_msg, "raw_score": 0.0}
                except Exception as e:
                    # 捕获其他类型的异常
                    import traceback
                    error_msg = str(e) if e and str(e) else "Unknown exception in _call_base"
                    error_detail = f"{error_msg}\n{traceback.format_exc()}"
                    print(f"[ERROR] _call_base 异常: {error_detail}", flush=True)
                    return {"success": False, "error": error_msg, "raw_score": 0.0}

            with ThreadPoolExecutor(max_workers=2) as executor:
                f_sam3 = executor.submit(_call_sam3)
                f_base = executor.submit(_call_base)
                try:
                    sam3_result = f_sam3.result()
                    base_result = f_base.result()
                except Exception as e:
                    # 如果线程执行过程中出现异常（不应该发生，但作为防御性检查）
                    return {"score": 0.0, "success": False, "error": f"Thread execution failed: {str(e)}"}

            # 检查sam3结果
            if not isinstance(sam3_result, dict):
                return {"score": 0.0, "success": False, "error": f"SAM3 result is not a dict: {type(sam3_result)}"}
            
            # _call_sam3() 正常返回时是 {"score": ..., "raw_score": ..., "reward_type": ...}
            # 异常时返回 {"success": False, "error": ..., "score": 0.0}
            # 检查是否有 success 字段且为 False（表示错误）
            if sam3_result.get("success") is False:
                # _call_sam3() 已经确保返回包含 error 字段的字典
                error_msg = sam3_result.get("error", "")
                if not error_msg:
                    error_msg = "SAM3 reward computation failed (no error message provided)"
                print(f"[ERROR] compute_stage2_reward_api: sam3_result 失败: {error_msg}", flush=True)
                return {"score": 0.0, "success": False, "error": str(error_msg)}
            
            # 检查base结果
            # RewardClient.compute_reward成功时返回{"score": ..., "raw_score": ..., "reward_type": ...}
            # 失败时会抛出RuntimeError，已被_call_base捕获并返回{"success": False, "error": ...}
            if not base_result:
                return {"score": 0.0, "success": False, "error": "Base reward result is empty"}
            
            if not isinstance(base_result, dict):
                return {"score": 0.0, "success": False, "error": f"Base result is not a dict: {type(base_result)}"}
            
            # 如果base_result有success字段且为False，说明调用失败
            if base_result.get("success") is False:
                error_msg = base_result.get("error", "")
                if not error_msg:
                    error_msg = "Base reward computation failed (no error message provided)"
                print(f"[ERROR] compute_stage2_reward_api: base_reward 调用失败: {error_msg}", flush=True)
                return {"score": 0.0, "success": False, "error": error_msg}

            # RewardClient.compute_reward成功时返回{"score": ..., "raw_score": ..., "reward_type": ...}
            # 检查sam3_result是否包含score字段
            if "score" not in sam3_result:
                return {"score": 0.0, "success": False, "error": "SAM3 reward result missing score field"}
            
            sam3_score = sam3_result.get("score", 0.0)
            
            # 检查base_result是否包含raw_score字段
            if "raw_score" not in base_result and "score" not in base_result:
                return {"score": 0.0, "success": False, "error": "Base reward result missing score/raw_score field"}
            
            base_score = base_result.get("raw_score", base_result.get("score", 0.0))
            score = 0.5 * sam3_score + 0.5 * base_score

            return {
                "score": score,
                "success": True,
                "sam3_score": sam3_score,
                "base_score": base_score,
                "edited_image": edited_image
            }
        else:
            # 根据 reward_type 获取正确的客户端（clip 或 omniverifier）
            target_reward_client = get_reward_client_for_type(reward_type)
            if target_reward_client is None:
                return {"score": 0.0, "success": False, "error": f"{reward_type} reward client not available"}
            
            reward_result = target_reward_client.compute_reward(
                image=edited_image,
                prompt=original_prompt,
                reward_type=reward_type,
            )
            score = reward_result["raw_score"]
        
        return {
            "score": score,
            "success": True,
            "edited_image": edited_image  # 可选
        }
        
    except Exception as e:
        # 确保错误消息不为空
        error_msg = str(e) if e else "Unknown exception in compute_stage2_reward_api"
        if not error_msg:
            error_msg = "Unknown exception in compute_stage2_reward_api"
        import traceback
        error_detail = f"{error_msg}\n{traceback.format_exc()}"
        print(f"[ERROR] compute_stage2_reward_api 异常: {error_detail[:500]}", flush=True)
        return {
            "score": 0.0,  # 编辑失败给予惩罚
            "success": False,
            "error": error_msg
        }


def compute_score(
    reward_inputs: list[dict[str, Any]],
    think_format_weight: float = 0.1,  # think格式奖励权重（在stage1中）
    json_format_weight: float = 0.1,  # json格式奖励权重（在stage2中，与edit任务一起）
    stage1_weight: float = 0.4,
    stage2_weight: float = 0.6,
    enable_stage2: bool = True,
    image_dir: Optional[str] = None,
    edit_config: Optional[dict] = None,
    max_workers: Optional[int] = None,  # 并行处理的线程数（默认为端点数量 × 2，自动计算）
    default_reward_type: Optional[str] = None,  # 默认奖励类型（clip或omniverifier），优先级最高
    virtual_correct_reward: float = 0.0,  # true+true时的虚拟奖励（默认0.0，不启用）
    # JSON格式检查配置
    require_edit_prompt_for_true: bool = True,  # 是否要求answer=true时也必须有edit_prompt字段
    **kwargs
) -> list[dict[str, float]]:
    """
    计算两阶段奖励（API版本，支持并行处理）
    
    与原版参数兼容，但使用API服务而不是本地模型
    第二阶段奖励会并行处理，充分利用多个GPU
    
    Args:
        max_workers: 并行处理的线程数。如果为None，则自动设置为端点数量 × 2（每个GPU 2个并发）
        default_reward_type: 默认奖励类型（clip或omniverifier）。如果设置，将覆盖数据中的reward_type和category映射。
                            优先级：default_reward_type > ground_truth中的reward_type > category映射 > "omniverifier"
                            也可以通过环境变量REWARD_TYPE设置
        
        image_dir: 图像目录路径，用于处理相对路径。
            - 作用机制：如果数据中的 image_path 是相对路径，会与 image_dir 拼接成完整路径
            - 判断：使用 os.path.isabs() 检查路径是否以 '/' 开头（绝对路径）
            - 示例：
              * 相对路径 "subdir/img.png" + image_dir -> "/base/dir/subdir/img.png"
              * 绝对路径 "/absolute/img.png" -> 直接使用，不拼接
            - 注意：当前数据（eval.json）中的 image_path 都是绝对路径，所以主要用于容错
            - 如果不传：相对路径将无法正确解析，可能导致文件找不到
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for omniverifier.")
    
    # 确定默认奖励类型（优先级：参数 > 环境变量 > None）
    if default_reward_type is None:
        default_reward_type = os.environ.get("REWARD_TYPE", None)
    
    # 验证奖励类型
    if default_reward_type is not None:
        if default_reward_type not in _SUPPORTED_REWARD_TYPES:
            # 优化：只在第一次打印，避免重复打印
            if not _client_warning_printed.get("default_reward_type", False):
                print(f"Warning: 不支持的奖励类型: {default_reward_type}，支持的类型: {_SUPPORTED_REWARD_TYPES}")
                print(f"  将使用默认值: clip")
                _client_warning_printed["default_reward_type"] = True
            default_reward_type = None
        else:
            # 优化：只在第一次打印，避免重复打印
            if not _client_warning_printed.get("default_reward_type", False):
                print(f"[INFO] 使用外部指定的奖励类型: {default_reward_type}")
                _client_warning_printed["default_reward_type"] = True
    
    # 初始化客户端
    # 注意：sam3_client 不需要在这里初始化，因为 compute_stage2_reward_api 会自己初始化
    try:
        edit_client, reward_client, _ = initialize_clients()
    except Exception as e:
        print(f"Warning: Failed to initialize API clients: {e}")
        print("Stage 2 rewards will be disabled.")
        enable_stage2 = False
        edit_client = None
        reward_client = None
    
    # 确定max_workers（如果为None，则根据端点数量自动计算）
    if max_workers is None:
        if edit_client is not None and len(edit_client.endpoints) > 0:
            # 优化：端点数量 × 1（而不是 × 2）
            # 原因：每个端点处理一个请求需要30-90秒，2倍并发可能导致请求堆积
            # 32个端点 → max_workers=32，可以充分利用所有端点，避免过度并发
            max_workers = len(edit_client.endpoints)
            # 优化：只在第一次打印，避免多个worker进程重复打印
            if not _client_warning_printed.get("max_workers", False):
                print(f"[INFO] 自动设置max_workers={max_workers} (端点数量={len(edit_client.endpoints)})")
                _client_warning_printed["max_workers"] = True
        else:
            # 如果没有edit_client，使用默认值1
            max_workers = 1
            # 优化：只在第一次打印，避免多个worker进程重复打印
            if not _client_warning_printed.get("max_workers", False):
                print(f"[INFO] 未检测到edit_client，使用默认值max_workers=1")
                _client_warning_printed["max_workers"] = True
    
    # 从kwargs中提取可能的额外字段
    batch_image_paths = kwargs.get("image_paths", None)
    batch_prompts = kwargs.get("prompts", None)
    
    # ========== 第一阶段：计算所有样本的第一阶段奖励 ==========
    stage1_results = []
    stage2_tasks = []  # 需要第二阶段奖励的任务列表
    
    for idx, reward_input in enumerate(reward_inputs):
        # 验证reward_input结构
        if not isinstance(reward_input, dict):
            print(f"Warning: reward_input[{idx}] is not a dict, skipping")
            continue
        if "response" not in reward_input:
            print(f"Warning: reward_input[{idx}] missing 'response' key, skipping")
            continue
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        
        # ========== 第一阶段奖励 ==========
        # 设计逻辑：think格式是前提，必须先有think标签，才能解析后面的内容
        # Stage1只包含think格式奖励和accuracy奖励
        think_score = think_format_reward(response)
        
        response_clean, _ = filter_thinking_part(response)
        accuracy_score = accuracy_reward(response_clean, reward_input["ground_truth"])
        
        # 计算第一阶段奖励
        # 权重分配：accuracy_weight + think_format_weight = 1.0
        # 设计逻辑：think格式不对 → 无法解析 → accuracy=0，所以模型必须先学think格式
        accuracy_weight = 1.0 - think_format_weight
        if accuracy_weight < 0:
            print(f"[WARNING] 权重分配不合理: think_format_weight={think_format_weight}, accuracy_weight={accuracy_weight}")
            print(f"  请确保 think_format_weight <= 1.0")
            accuracy_weight = max(0.0, accuracy_weight)  # 至少为0
        
        stage1_reward = (
            accuracy_weight * accuracy_score + 
            think_format_weight * think_score
        )
        
        # 计算json格式奖励（用于stage2，与edit任务一起）
        # 注意：json格式奖励主要目的是确保能提取edit_prompt，所以只在answer=false时计算
        json_score = json_format_reward(response, require_edit_prompt_for_true=require_edit_prompt_for_true)
        
        # 判断模型回答的answer值（用于决定是否计算json格式奖励）
        model_answer = None
        try:
            model_answer_match = re.search(r'"answer"\s*:\s*(true|false)', response_clean, re.IGNORECASE)
            if model_answer_match:
                model_answer = model_answer_match.group(1).lower() == "true"
        except (AttributeError, IndexError):
            model_answer = False
        
        # 保存第一阶段结果
        stage1_results.append({
            "idx": idx,
            "think_score": think_score,
            "json_score": json_score,  # 保存json_score，用于stage2计算
            "model_answer": model_answer,  # 保存model_answer，用于判断是否计算json格式奖励
            "accuracy_score": accuracy_score,
            "stage1_reward": stage1_reward,
            "response_clean": response_clean,
        })
        
        # ========== 准备第二阶段奖励任务 ==========
        stage2_task = None
        
        if enable_stage2:
            try:
                if "ground_truth" not in reward_input:
                    print(f"Warning: reward_input[{idx}] missing 'ground_truth' key, skipping stage2")
                    continue
                gt_data = json.loads(reward_input["ground_truth"])
                # 验证gt_data是字典类型
                if not isinstance(gt_data, dict):
                    print(f"Warning: reward_input[{idx}] ground_truth is not a dict, skipping stage2")
                    continue
                gt_answer = gt_data.get("answer", False)
                
                category = gt_data.get("category", "")
                # 确定奖励类型（优先级：外部参数 > ground_truth中的reward_type > category映射 > "omniverifier"）
                if default_reward_type is not None:
                    # 如果设置了外部默认奖励类型，优先使用
                    reward_type = default_reward_type
                else:
                    # 否则从ground_truth中获取
                    reward_type = gt_data.get("reward_type")
                    if not reward_type:
                        # 如果ground_truth中没有，使用category映射
                        reward_type = _CATEGORY_TO_REWARD_TYPE.get(category, "omniverifier")
                
                model_answer_match = re.search(r'"answer"\s*:\s*(true|false)', response_clean, re.IGNORECASE)
                model_answer = False
                if model_answer_match:
                    try:
                        model_answer = model_answer_match.group(1).lower() == "true"
                    except (AttributeError, IndexError):
                        model_answer = False
                
                # 只有当模型判断为false且真实标签也为false时，才计算第二阶段奖励
                if not model_answer and not gt_answer:
                    # 提取edit_prompt和explanation（与bagel_editor.py格式一致）
                    edit_info = extract_edit_info(response)
                    
                    if edit_info and edit_info.get("edit_prompt"):
                        edit_prompt = edit_info.get("edit_prompt")
                        explanation = edit_info.get("explanation", "")
                        
                        # 获取图像路径和prompt
                        # 优先级：1. reward_input中的images字段（与image_dir拼接） 2. batch_image_paths 3. reward_input中的image_path 4. gt_data中的image_path（最后的后备选项）
                        image_path = ""
                        # 优先使用reward_input中的images字段（这是训练数据中实际使用的图片路径）
                        if "images" in reward_input and isinstance(reward_input["images"], list) and len(reward_input["images"]) > 0:
                            image_path = reward_input["images"][0]
                        # 如果没有images字段，尝试batch_image_paths
                        if not image_path and batch_image_paths is not None and idx < len(batch_image_paths):
                            image_path = batch_image_paths[idx]
                        # 如果还没有，尝试reward_input中的image_path
                        if not image_path:
                            image_path = reward_input.get("image_path", "")
                        # 最后的后备选项：使用gt_data中的image_path（但这不是推荐的方式，因为可能指向错误的路径）
                        if not image_path:
                            image_path = gt_data.get("image_path", "")
                        
                        # 获取prompt
                        prompt = reward_input.get("prompt", "")
                        if not prompt:
                            prompt = gt_data.get("prompt", "")
                        if not prompt and batch_prompts is not None and idx < len(batch_prompts):
                            prompt = batch_prompts[idx]
                        
                        # 处理相对路径：如果 image_path 是相对路径，则与 image_dir 拼接
                        # 判断逻辑：使用 os.path.isabs() 检查路径是否以 '/' 开头（绝对路径）
                        # - 绝对路径：直接使用，不拼接（如 "/data/images/img.png"）
                        # - 相对路径：与 image_dir 拼接（如 "subdir/img.png" -> "/base/dir/subdir/img.png"）
                        # 如果不传 image_dir 或 image_path 已经是绝对路径，则不会拼接
                        if image_path and image_dir and not os.path.isabs(image_path):
                            image_path = os.path.join(image_dir, image_path)
                        
                        if image_path and os.path.exists(image_path) and os.path.isfile(image_path) and prompt:
                            # 准备第二阶段任务（稍后并行处理）
                            # 保存json_score，用于stage2计算
                            stage2_task = {
                                "idx": idx,
                                "edit_prompt": edit_prompt,
                                "explanation": explanation,  # 添加explanation字段
                                "image_path": image_path,
                                "prompt": prompt,
                                "reward_type": reward_type,
                                "edit_config": edit_config,
                                "json_score": json_score,  # 保存json格式分数，用于stage2计算
                                "ground_truth": reward_input["ground_truth"],  # 传递ground_truth，SAM3奖励计算需要
                                "is_false_false": True,  # 标记这是false+false的情况
                            }
                        else:
                            stage2_task = {
                                "idx": idx,
                                "error": "Missing image_path or prompt",
                                "stage2_reward": 0.0,
                                "json_score": json_score,  # 保存json_score，用于stage2计算（answer=false时需要）
                                "is_false_false": True,  # 标记这是false+false的情况
                            }
                    else:
                        stage2_task = {
                            "idx": idx,
                            "error": "No edit_prompt extracted",
                            "stage2_reward": 0.0,
                            "json_score": json_score,  # 保存json_score，用于stage2计算（answer=false时需要）
                            "is_false_false": True,  # 标记这是false+false的情况
                        }
                elif model_answer and gt_answer:
                    # true+true时，如果virtual_correct_reward > 0，则给虚拟奖励
                    stage2_reward = (1 - json_format_weight) * virtual_correct_reward + json_format_weight * json_score
                    # NOTE:
                    # 如果TP不需要计算json奖励的时候使用
                    # stage2_reward = virtual_correct_reward

                    stage2_task = {
                        "idx": idx,
                        "json_score": json_score,
                        "stage2_reward": stage2_reward,
                        "stage2_details": {
                            "type": "TP",
                            "json_score": json_score,
                        },
                    }
                elif model_answer and not gt_answer:
                    stage2_task = {
                        "idx": idx,
                        "stage2_reward": 0.0,
                        "stage2_details": {
                            "type": "FP",
                            "json_score": 0.0,  # 判断错误，不计算json格式奖励
                        },
                    }
                elif not model_answer and gt_answer:
                    stage2_task = {
                        "idx": idx,
                        "stage2_reward": 0.0,
                        "stage2_details": {
                            "type": "FN",
                            "json_score": 0.0,  # 判断错误，不计算json格式奖励
                        },
                    }
   
            except Exception as e:
                # 解析ground_truth失败：无法判断answer，不计算json格式奖励
                # 注意：此时无法判断model_answer和gt_answer，所以json_score=0.0
                stage2_task = {
                    "idx": idx,
                    "error": f"Failed to parse ground_truth (data issue): {e}",
                    "stage2_reward": 0.0,  # 保持0.0，不惩罚
                    "stage2_details": {"type": "error", "reason": f"Parse error: {e}"},
                    "json_score": 0.0,  # 无法判断answer，不计算json格式奖励
                }
        
        if stage2_task is not None:
            stage2_tasks.append(stage2_task)
    
    # ========== 第二阶段：并行处理所有需要第二阶段奖励的任务 ==========
    stage2_results = {}  # idx -> {stage2_reward, stage2_details}
    
    # 注意：奖励客户端在 compute_stage2_reward_api 内部会根据 reward_type 自动创建
    # 在 sam3 和 mixed 模式下，不需要外部的 reward_client
    # 只有在 clip/omniverifier 模式下，才可能需要外部的 reward_client
    # 但即使 reward_client 为 None，compute_stage2_reward_api 内部也会尝试创建相应的客户端
    # 所以这里只需要检查 edit_client 是否存在即可
    if stage2_tasks and enable_stage2 and edit_client is not None:

        def process_stage2_task(task: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
            """处理单个第二阶段奖励任务"""
            idx = task["idx"]
            
            # 如果已经有错误或预设的奖励，需要加上json格式奖励
            if "error" in task or "stage2_reward" in task:
                stage2_details = task.get("stage2_details", {})
                if not stage2_details or "type" not in stage2_details:
                    # 如果没有stage2_details，根据error信息创建
                    stage2_details = {
                        "type": "error",
                        "reason": task.get("error", "Unknown error")
                    }
                # 保留is_false_false标记
                if task.get("is_false_false", False):
                    stage2_details["is_false_false"] = True
                
                # 获取json_score（从task中获取，已经在准备stage2_task时根据情况设置）
                stage2_details["json_score"] = task.get("json_score", 0.0)
                stage2_reward = task.get("stage2_reward", 0.0)
                
                return idx, {
                    "stage2_reward": stage2_reward,
                    "stage2_details": stage2_details,
                }
            
            # 调用API计算第二阶段奖励
            try:
                stage2_result = compute_stage2_reward_api(
                    edit_prompt=task["edit_prompt"],
                    original_image_path=task["image_path"],
                    original_prompt=task["prompt"],
                    reward_type=task["reward_type"],
                    edit_config=task["edit_config"],
                    explanation=task.get("explanation"),  # 传递explanation参数
                    ground_truth=task.get("ground_truth"),  # 传递ground_truth，SAM3奖励计算需要
                )
                
                # 检查是否成功，如果失败则返回错误信息
                if not stage2_result.get("success", False):
                    error_msg = stage2_result.get("error", "")
                    if not error_msg:
                        error_msg = "Stage2 reward computation failed (no error message provided)"
                    json_score = task.get("json_score", 0.0)
                    stage2_reward = json_format_weight * json_score
                    return idx, {
                        "stage2_reward": stage2_reward,
                        "stage2_details": {
                            "type": "error",
                            "reason": error_msg,
                            "json_score": json_score,
                            "is_false_false": task.get("is_false_false", False),
                        },
                    }
                
                # 直接使用分数（GRPO的组内归一化会处理分数差异）
                edit_reward = stage2_result["score"]
                
                # Stage2奖励 = json格式奖励 + edit任务奖励
                # 设计逻辑：json格式和edit_prompt相关，如果json格式不对，无法提取edit_prompt
                # 权重分配：json_format_weight + edit_reward_weight = 1.0
                edit_reward_weight = 1.0 - json_format_weight
                if edit_reward_weight < 0:
                    edit_reward_weight = max(0.0, edit_reward_weight)
                
                # 获取json_score（从stage1_results中获取）
                json_score = task.get("json_score", 0.0)
                
                # 计算stage2奖励（json格式奖励 + edit任务奖励）
                stage2_reward = (
                    json_format_weight * json_score +
                    edit_reward_weight * edit_reward
                )
                
                # 确定显示类型：如果是false+false的情况（通过图像编辑计算），统一显示为TN
                # 无论使用什么奖励类型（clip、omniverifier、sam3、mixed），都应该归类为TN
                display_type = "TN" if task.get("is_false_false", False) else task["reward_type"]
                
                # 成功时，error 字段应该是 None 或不存在，不需要保存
                stage2_details = {
                    "type": display_type,  # 统一显示为TN（如果是false+false）或原始reward_type
                    "reward_type": task["reward_type"],  # 保留原始reward_type用于调试
                    "success": True,  # 成功时明确设置为 True
                    "edit_reward": edit_reward,  # 保存原始edit奖励
                    "json_score": json_score,  # 保存json格式分数
                    "is_false_false": task.get("is_false_false", False),  # 保留false+false标记
                }
                # 只有在有错误信息时才添加 error 字段
                if stage2_result.get("error"):
                    stage2_details["error"] = stage2_result.get("error")
                
                return idx, {
                    "stage2_reward": stage2_reward,
                    "stage2_details": stage2_details,
                }
            except Exception as e:
                # 即使出错，也要计算json格式奖励
                json_score = task.get("json_score", 0.0)
                stage2_reward = json_format_weight * json_score
                return idx, {
                    "stage2_reward": stage2_reward,
                    "stage2_details": {
                        "type": "error",
                        "reason": str(e),
                        "json_score": json_score,
                    },
                }
        
        # 并行处理所有第二阶段任务
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(process_stage2_task, task): task for task in stage2_tasks}
            
            for future in as_completed(future_to_task):
                try:
                    idx, result = future.result()
                    stage2_results[idx] = result
                except Exception as e:
                    task = future_to_task[future]
                    idx = task["idx"]
                    stage2_results[idx] = {
                        "stage2_reward": 0.0,
                        "stage2_details": {"type": "error", "reason": f"Task execution failed: {e}"},
                    }
    
    # ========== 合并结果 ==========
    scores = []
    for idx, stage1_result in enumerate(stage1_results):
        # 获取第二阶段结果（如果没有，使用默认值）
        model_answer = stage1_result.get("model_answer")
        stage2_result = stage2_results.get(idx, {
            "stage2_reward": 0.0,
            "stage2_details": {"type": "none"}
        })
        
        # 计算总奖励
        overall_reward = (
            stage1_weight * stage1_result["stage1_reward"] +
            stage2_weight * stage2_result["stage2_reward"]
        )
        
        stage2_edit_reward = stage2_result["stage2_details"].get("edit_reward", 0.0)
        is_false_false = stage2_result["stage2_details"].get("is_false_false", False)
        
        scores.append({
            "overall": overall_reward, # 最终得分，DAPO根据这个字段来进行优化
            "stage1_reward": stage1_result["stage1_reward"],
            "stage2_reward": stage2_result["stage2_reward"],
            "think_format": stage1_result["think_score"],
            "json_format": stage1_result["json_score"],  # json格式奖励在stage2中计算，这里保留用于统计
            "stage1_accuracy": stage1_result["accuracy_score"],
            "stage2_accuracy": stage2_edit_reward, 
            "TN_counts": 1.0 if is_false_false else 0.0,
            # stage2_details 包含完整的类型信息（dict类型，不会被wandb记录，但会保存到JSON文件）
            "stage2_details": stage2_result["stage2_details"]
        })
    
    # 统计并打印stage2奖励类型分布（训练和验证时都显示，方便实时监控）
    stage2_type_counts = {}
    # 细化统计：详细分类的计数和平均奖励
    stage2_stats = {
        "virtual_correct_count": 0,  # 都是true，虚拟奖励
        "virtual_correct_reward_sum": 0.0,
        "computed_count": 0,  # 都是false，通过图像编辑计算（重点关注）
        "computed_reward_sum": 0.0,
        "error_count": 0,  # error情况（无法提取edit_prompt或其他错误）
        "error_reasons": {},  # error的具体原因统计
        "virtual_neutral_fp_count": 0,  # False positive: 模型true，真实false
        "virtual_neutral_fn_count": 0,  # False negative: 模型false，真实true
        # false+false情况的详细统计（包括成功计算的和error的）
        "false_false_count": 0,  # false+false总数量
        "false_false_computed_count": 0,  # false+false中成功计算的
        "false_false_computed_rewards": [],  # false+false中成功计算的奖励值列表（用于计算分位数）
        "false_false_error_count": 0,  # false+false中error的数量
        "false_false_error_reasons": {},  # false+false中error的具体原因
    }
    
    for score in scores:
        stage2_type = score.get("stage2_details", {}).get("type", "none")
        stage2_type_counts[stage2_type] = stage2_type_counts.get(stage2_type, 0) + 1
        
        stage2_reward = score.get("stage2_reward", 0.0)
        stage2_details = score.get("stage2_details", {})
        stage2_reason = stage2_details.get("reason", "")
        stage2_error = stage2_details.get("error")
        
        # 检查是否是false+false的情况
        is_false_false = stage2_details.get("is_false_false", False)
        
        # 细化统计各种情况
        if stage2_type == "TP":
            # True Positive: 模型true，真实true
            stage2_stats["virtual_correct_count"] += 1
            stage2_stats["virtual_correct_reward_sum"] += stage2_reward
        elif stage2_type == "TN" or stage2_type in ["clip", "omniverifier", "sam3", "mixed"]:
            # True Negative: 模型false，真实false，通过图像编辑计算
            # 统一处理：TN（已统一命名）、clip、omniverifier、sam3、mixed
            stage2_stats["computed_count"] += 1
            stage2_stats["computed_reward_sum"] += stage2_reward
            # 如果是false+false的情况，单独统计
            if is_false_false:
                stage2_stats["false_false_computed_count"] += 1
                stage2_stats["false_false_computed_rewards"].append(stage2_reward)
        elif stage2_type == "error":
            stage2_stats["error_count"] += 1
            # 统计error的具体原因
            error_reason = stage2_reason or "Unknown error"
            if stage2_error:
                error_reason = f"{error_reason} ({stage2_error[:50]})"  # 截断错误信息，避免过长
            stage2_stats["error_reasons"][error_reason] = stage2_stats["error_reasons"].get(error_reason, 0) + 1
            # 如果是false+false的情况，单独统计
            if is_false_false:
                stage2_stats["false_false_error_count"] += 1
                stage2_stats["false_false_error_reasons"][error_reason] = stage2_stats["false_false_error_reasons"].get(error_reason, 0) + 1
        elif stage2_type == "FP":
            # False Positive: 模型true，真实false
            stage2_stats["virtual_neutral_fp_count"] += 1
        elif stage2_type == "FN":
            # False Negative: 模型false，真实true
            stage2_stats["virtual_neutral_fn_count"] += 1

        
        # 统计false+false的总数
        if is_false_false:
            stage2_stats["false_false_count"] += 1
    
    if stage2_type_counts:
        batch_size = len(scores)
        total = sum(stage2_type_counts.values())
        
        # 统一命名：将所有通过图像编辑计算的类型（clip、omniverifier、sam3、mixed）合并为 TN 显示
        # 注意：如果type已经是TN（在process_stage2_task中已设置），直接使用
        unified_type_counts = {}
        tn_count = 0
        for k, v in stage2_type_counts.items():
            if k == "TN":
                # 如果已经是TN，直接累加
                tn_count += v
            elif k in ["omniverifier", "clip", "sam3", "mixed"]:
                # 将这些类型合并为TN（向后兼容，防止某些情况下type未正确设置）
                tn_count += v
            else:
                unified_type_counts[k] = v
        if tn_count > 0:
            unified_type_counts["TN"] = tn_count
        
        # 打印总体统计（使用统一命名）
        print(f"[INFO] Stage2奖励类型统计 (batch_size={batch_size}): {dict(sorted(unified_type_counts.items(), key=lambda x: x[1], reverse=True))}")
        print(f"[INFO] Stage2奖励类型占比: {', '.join([f'{k}: {v/total*100:.1f}%' for k, v in sorted(unified_type_counts.items(), key=lambda x: x[1], reverse=True)])}")
        print()
        
        # 打印详细分类统计
        if stage2_stats["virtual_correct_count"] > 0:
            avg_virtual = stage2_stats["virtual_correct_reward_sum"] / stage2_stats["virtual_correct_count"]
            if virtual_correct_reward > 0:
                # 当virtual_correct_reward>0时，stage2_reward = (1-json_format_weight)*virtual_correct_reward + json_format_weight*json_score
                # avg_virtual是stage2_reward的平均值，包含虚拟奖励和json格式奖励两部分
                print(f"[INFO] Stage2详细分类 - TP (模型true，真实true，stage2奖励=(1-json_format_weight)×虚拟奖励{virtual_correct_reward} + json_format_weight×json_score): {stage2_stats['virtual_correct_count']}个 ({stage2_stats['virtual_correct_count']/total*100:.1f}%), 平均stage2奖励: {avg_virtual:.3f}")
            else:
                # 当virtual_correct_reward=0时，stage2_reward = json_format_weight * json_score
                # avg_virtual是stage2_reward的平均值，即json_format_weight加权后的json_score平均值
                print(f"[INFO] Stage2详细分类 - TP (模型true，真实true，stage2奖励=json_format_weight×json_score): {stage2_stats['virtual_correct_count']}个 ({stage2_stats['virtual_correct_count']/total*100:.1f}%), 平均stage2奖励: {avg_virtual:.3f}")
        
        # 详细统计TN的情况（包括成功计算的和error的）
        if stage2_stats["false_false_count"] > 0:
            print(f"[INFO] Stage2详细分类 - TN (模型false，真实false，需要编辑): {stage2_stats['false_false_count']}个 ({stage2_stats['false_false_count']/total*100:.1f}%)")
            
            # 成功计算的情况（平均奖励基于成功计算的样本）
            if stage2_stats["false_false_computed_count"] > 0:
                false_false_rewards = stage2_stats["false_false_computed_rewards"]
                avg_false_false = sum(false_false_rewards) / len(false_false_rewards)
                
                print(f"      ✅ 成功计算编辑奖励: {stage2_stats['false_false_computed_count']}个")
                print(f"         - 平均奖励: {avg_false_false:.3f}")
  
            
            # error的情况
            if stage2_stats["false_false_error_count"] > 0:
                print(f"      ❌ 无法计算编辑奖励 (Error): {stage2_stats['false_false_error_count']}个")
                for reason, count in sorted(stage2_stats["false_false_error_reasons"].items(), key=lambda x: x[1], reverse=True):
                    print(f"         • {reason}: {count}个")
        
        # 详细显示非TN的error情况（例如解析ground_truth失败等）
        non_tn_error_count = stage2_stats["error_count"] - stage2_stats["false_false_error_count"]
        if non_tn_error_count > 0:
            print(f"[INFO] Stage2详细分类 - Error (非TN的error，无法计算Stage2奖励): {non_tn_error_count}个 ({non_tn_error_count/total*100:.1f}%)")
            # 只显示非TN的error原因
            non_tn_error_reasons = {}
            for reason, count in stage2_stats["error_reasons"].items():
                tn_error_count = stage2_stats["false_false_error_reasons"].get(reason, 0)
                if count > tn_error_count:
                    non_tn_error_reasons[reason] = count - tn_error_count
            for reason, count in sorted(non_tn_error_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"      • {reason}: {count}个")
        
        # 详细显示FP和FN情况
        virtual_neutral_total = stage2_stats["virtual_neutral_fp_count"] + stage2_stats["virtual_neutral_fn_count"]
        if virtual_neutral_total > 0:
            print(f"[INFO] Stage2详细分类 - 判断错误 (模型和真实答案不一致，stage2_reward=0.0): {virtual_neutral_total}个 ({virtual_neutral_total/total*100:.1f}%)")
            if stage2_stats["virtual_neutral_fp_count"] > 0:
                print(f"      • FP (False Positive，模型true，真实false): {stage2_stats['virtual_neutral_fp_count']}个")
            if stage2_stats["virtual_neutral_fn_count"] > 0:
                print(f"      • FN (False Negative，模型false，真实true): {stage2_stats['virtual_neutral_fn_count']}个")
    
    return scores