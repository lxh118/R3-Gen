#!/usr/bin/env python3
"""
OmniVerifier奖励服务 - 使用OmniVerifier-7B模型进行奖励计算
支持多GPU负载均衡
"""

import argparse
import io
import base64
import os
import sys
import json
import re
import logging

import torch
from flask import Flask, request, jsonify
from PIL import Image

# 导入OmniVerifier模型
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    OMNIVERIFIER_AVAILABLE = True
except ImportError:
    OMNIVERIFIER_AVAILABLE = False
    print("Warning: OmniVerifier dependencies not available. Please install transformers and qwen-vl-utils")

app = Flask(__name__)

# 全局模型变量
OMNIVERIFIER_MODEL = None
OMNIVERIFIER_PROCESSOR = None
DEVICE = None


def load_omniverifier_model(model_path: str, device: torch.device):
    """加载OmniVerifier模型"""
    global OMNIVERIFIER_MODEL, OMNIVERIFIER_PROCESSOR
    
    if not OMNIVERIFIER_AVAILABLE:
        raise RuntimeError("OmniVerifier dependencies not available. Please install transformers and qwen-vl-utils")
    
    print(f"Loading OmniVerifier model from {model_path}...")
    
    try:
        # 加载模型（使用bfloat16精度，明确指定单个设备，避免多GPU分布）
        # 不使用device_map="auto"，因为它会将模型分布到多个GPU，导致设备不匹配错误
        # 直接加载到CPU，然后手动移动到指定设备，这样更可控
        OMNIVERIFIER_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=None,  # 不使用自动设备映射
            trust_remote_code=True
        )
        
        # 手动移动到指定设备（确保整个模型在同一个设备上）
        OMNIVERIFIER_MODEL = OMNIVERIFIER_MODEL.to(device)
        
        # 加载processor
        OMNIVERIFIER_PROCESSOR = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print(f"✅ OmniVerifier model loaded successfully")
        print(f"   Model path: {model_path}")
        print(f"   Device: {device}")
        print(f"   Model dtype: {OMNIVERIFIER_MODEL.dtype if hasattr(OMNIVERIFIER_MODEL, 'dtype') else 'N/A'}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load OmniVerifier model: {e}")


def build_verification_prompt(original_prompt: str) -> str:
    """构建验证prompt"""
    question = f"""This image was generated from the prompt: {original_prompt}. 
    Please carefully analyze the image and determine whether all the objects, attributes, count, and spatial relationships mentioned in the prompt are correctly represented in the image. 

    If the image accurately reflects the prompt, please answer 'true'; otherwise, answer 'false'.  

    Respond strictly in the following JSON format: 

    {{
        "answer": true/false,
        "explanation": "If the answer is false, briefly summarize the main error.",
    }}

 You should first think about the reasoning process in your mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"""

    return question


def build_question_prompt(question: str) -> str:
    """构建单个问题的验证prompt"""
    prompt = f"""Please carefully analyze the image and answer the following question: {question}

    If the image accurately reflects the prompt, please answer 'true'; otherwise, answer 'false'. 

    Respond strictly in the following JSON format:

    {{
        "answer": true/false,
        "explanation": "Brief explanation of your answer.",
    }}

You should first think about the reasoning process in your mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"""

    return prompt


# def filter_thinking_part(response: str, eos_token=None) -> tuple[str, bool]:
#     """
#     提取answer部分（去除think标签）
    
#     支持两种格式：
#     1. 标准格式：<think>think内容</think>answer
#     2. 只有结束标签：think内容</think>answer（模型偏好格式）
    
#     Returns:
#         (filtered_response: str, success: bool)
#     """
#     response_start = 0
#     success = False
#     think_tag_start = '<think>'
#     think_tag_end = '</think>'
    
#     # 查找结束标签（从后往前找最后一个）
#     think_end = response.rfind(think_tag_end)
    
#     if think_end != -1:
#         # 找到了结束标签，提取结束标签之后的内容
#         response_start = think_end + len(think_tag_end)
#         success = True
        
#         # 可选：检查是否有开始标签（用于兼容标准格式）
#         think_start = response.find(think_tag_start, 0, think_end)
#         # 如果有开始标签且位置正确，说明是标准格式；否则是只有结束标签的格式
#         # 两种格式都支持，所以这里不需要额外处理
#     else:
#         # 没有找到结束标签，尝试查找开始标签（向后兼容）
#         think_start = response.find(think_tag_start)
#         if think_start != -1:
#             # 有开始标签但没有结束标签，尝试从开始标签之后提取
#             response_start = think_start + len(think_tag_start)
#             success = True
    
#     if eos_token is not None:
#         response_end = response.find(eos_token, response_start)
#     else:
#         response_end = len(response)
#     response = response[response_start:response_end]
#     return response, success

def find_first_json_block(text: str) -> str | None:
    """
    在文本中查找第一个有效的 JSON 代码块（Robust Extraction）。
    支持 ```json ... ```, ``` ... ``` 和 纯 {...}
    """
    if not text:
        return None
        
    # 1. 尝试找 markdown 代码块
    markdown_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
    match = markdown_pattern.search(text)
    if match:
        return match.group(1)
    
    # 2. 尝试找最外层的括号 {...}
    start_idx = text.find('{')
    if start_idx == -1:
        return None
        
    brace_count = 0
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start_idx : i+1]
    
    return None

def extract_answer_from_response(response_text: str) -> tuple[bool, str]:
    """
    从模型响应中提取answer（支持think标签和更鲁棒的JSON提取）
    
    Returns:
        (answer: bool, explanation: str)
    """
    try:
        # 1. 尝试提取并解析 JSON (最优先，最鲁棒)
        json_str = find_first_json_block(response_text)
        
        if json_str:
            try:
                response_json = json.loads(json_str)
                if isinstance(response_json, dict):
                    # 处理 answer 字段
                    answer_val = response_json.get("answer", False)
                    if isinstance(answer_val, str):
                        answer = answer_val.lower() == "true"
                    else:
                        answer = bool(answer_val)
                        
                    explanation = response_json.get("explanation", "")
                    return answer, explanation
            except (json.JSONDecodeError, ValueError, TypeError):
                # JSON 解析失败，继续尝试正则兜底
                pass
        
        # 2. 正则兜底 (Fallback)
        # 如果 JSON 结构不完整（例如少了大括号），正则可能还能救回来
        # 匹配 "answer": true/false，忽略大小写和空白
        answer_match = re.search(r'"answer"\s*:\s*(true|false)', response_text, re.IGNORECASE)
        if answer_match:
            answer_str = answer_match.group(1).lower()
            answer = answer_str == "true"
            
            # 尝试提取 explanation
            # 匹配 "explanation": "..."
            explanation_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', response_text, re.IGNORECASE)
            explanation = explanation_match.group(1) if explanation_match else ""
            
            return answer, explanation
            
        # 3. 终极兜底：直接找单词 "true" 或 "false" (仅在非常绝望时使用，容易误判，可选)
        # 考虑到 OmniVerifier 的 Prompt 强约束了 JSON，通常不需要这一步
        
        return False, "Failed to parse response"
        
    except Exception as e:
        return False, f"Exception in parsing: {str(e)}"

# def extract_answer_from_response(response_text: str) -> tuple[bool, str]:
#     """
#     从模型响应中提取answer（支持think标签和更鲁棒的JSON提取）
    
#     策略：
#     1. 先从 `</think>` 之后提取JSON（标准位置）
#     2. 如果失败，尝试从整个响应中查找JSON块
#     3. 如果还失败，尝试从think内容中提取JSON
#     4. 最后尝试正则表达式提取
    
#     Returns:
#         (answer: bool, explanation: str)
#     """
#     try:
#         # 策略1：尝试从 `</think>` 之后提取JSON（标准位置）
#         response_clean, has_think_tag = filter_thinking_part(response_text)
        
#         # 尝试解析JSON
#         try:
#             response_json = json.loads(response_clean.strip())
#             if isinstance(response_json, dict):
#                 answer = response_json.get("answer", False)
#                 explanation = response_json.get("explanation", "")
#                 return bool(answer), explanation
#         except (json.JSONDecodeError, ValueError, TypeError):
#             pass
        
#         # 策略2：如果策略1失败，尝试从整个响应中查找JSON块
#         # 查找JSON的开始和结束位置（使用大括号匹配）
#         json_start = response_text.find('{')
#         if json_start != -1:
#             # 从第一个 { 开始，尝试找到匹配的 }
#             brace_count = 0
#             json_end = -1
#             for i in range(json_start, len(response_text)):
#                 if response_text[i] == '{':
#                     brace_count += 1
#                 elif response_text[i] == '}':
#                     brace_count -= 1
#                     if brace_count == 0:
#                         json_end = i + 1
#                         break
            
#             if json_end != -1:
#                 try:
#                     json_str = response_text[json_start:json_end]
#                     response_json = json.loads(json_str)
#                     if isinstance(response_json, dict):
#                         answer = response_json.get("answer", False)
#                         explanation = response_json.get("explanation", "")
#                         return bool(answer), explanation
#                 except (json.JSONDecodeError, ValueError, TypeError):
#                     pass
        
#         # 策略3：如果策略2也失败，尝试从think内容中提取JSON
#         # 查找think内容（在 `</think>` 之前）
#         think_tag_end = '</think>'
#         think_end = response_text.rfind(think_tag_end)
#         if think_end != -1:
#             think_content = response_text[:think_end]
#             # 在think内容中查找JSON
#             json_start = think_content.find('{')
#             if json_start != -1:
#                 brace_count = 0
#                 json_end = -1
#                 for i in range(json_start, len(think_content)):
#                     if think_content[i] == '{':
#                         brace_count += 1
#                     elif think_content[i] == '}':
#                         brace_count -= 1
#                         if brace_count == 0:
#                             json_end = i + 1
#                             break
                
#                 if json_end != -1:
#                     try:
#                         json_str = think_content[json_start:json_end]
#                         response_json = json.loads(json_str)
#                         if isinstance(response_json, dict):
#                             answer = response_json.get("answer", False)
#                             explanation = response_json.get("explanation", "")
#                             return bool(answer), explanation
#                     except (json.JSONDecodeError, ValueError, TypeError):
#                         pass
        
#         # 策略4：最后尝试正则表达式提取（兜底方案）
#         answer_match = re.search(r'"answer"\s*:\s*(true|false)', response_text, re.IGNORECASE)
#         if answer_match:
#             answer_str = answer_match.group(1).lower()
#             answer = answer_str == "true"
            
#             # 尝试提取explanation
#             explanation_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', response_text, re.IGNORECASE)
#             explanation = explanation_match.group(1) if explanation_match else ""
            
#             return answer, explanation
        
#         # 如果所有策略都失败了，返回False
#         return False, "Failed to parse response"
        
#     except Exception as e:
#         return False, f"Exception in parsing: {str(e)}"


def compute_omniverifier_score(image: Image.Image, prompt: str) -> float:
    """
    使用OmniVerifier模型计算奖励分数
    
    Args:
        image: PIL图像
        prompt: 原始prompt
    
    Returns:
        奖励分数：如果answer为true返回1.0，否则返回0.0
    """
    global OMNIVERIFIER_MODEL, OMNIVERIFIER_PROCESSOR, DEVICE
    
    if OMNIVERIFIER_MODEL is None or OMNIVERIFIER_PROCESSOR is None:
        raise RuntimeError("OmniVerifier model not loaded")
    
    try:
        # 构建验证prompt
        question = build_verification_prompt(prompt)
        
        # 准备消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,  # 直接传递PIL图像
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        # 准备输入
        text = OMNIVERIFIER_PROCESSOR.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = OMNIVERIFIER_PROCESSOR(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # 移动到正确的设备
        # 优先使用全局DEVICE变量（启动时指定的设备）
        if DEVICE is not None:
            target_device = DEVICE
        elif hasattr(OMNIVERIFIER_MODEL, "device"):
            target_device = OMNIVERIFIER_MODEL.device
        else:
            # 获取模型第一个参数所在的设备
            target_device = next(OMNIVERIFIER_MODEL.parameters()).device
        
        # 确保所有输入都在同一个设备上
        inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # 推理（增加max_new_tokens以容纳think标签内容）
        with torch.no_grad():
            generated_ids = OMNIVERIFIER_MODEL.generate(**inputs, max_new_tokens=1024, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            output_text = OMNIVERIFIER_PROCESSOR.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        # 提取answer
        response_text = output_text[0] if output_text else ""
        answer, explanation = extract_answer_from_response(response_text)
        
        # 返回分数：true=1.0, false=0.0
        score = 1.0 if answer else 0.0
            
        return score
        
    except Exception as e:
        print(f"Error computing OmniVerifier score: {e}", flush=True)
        raise


def compute_omniverifier_score_with_qa(image: Image.Image, yn_question_list: list[str]) -> float:
    """
    使用OmniVerifier模型基于多个问题计算奖励分数
    
    Args:
        image: PIL图像
        yn_question_list: 是/否问题列表，例如 ["Is there a cup in the image?", "Is the cup red in color?"]
    
    Returns:
        奖励分数：正确答案数 / 总问题数，范围 [0.0, 1.0]
    """
    global OMNIVERIFIER_MODEL, OMNIVERIFIER_PROCESSOR, DEVICE
    
    if OMNIVERIFIER_MODEL is None or OMNIVERIFIER_PROCESSOR is None:
        raise RuntimeError("OmniVerifier model not loaded")
    
    if not yn_question_list:
        return 0.0
    
    correct_count = 0
    total_questions = len(yn_question_list)
    
    try:
        # 对每个问题单独判断
        for question in yn_question_list:
            # 构建问题prompt
            question_prompt = build_question_prompt(question)
            
            # 准备消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,  # 直接传递PIL图像
                        },
                        {"type": "text", "text": question_prompt},
                    ],
                }
            ]
            
            # 准备输入
            text = OMNIVERIFIER_PROCESSOR.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = OMNIVERIFIER_PROCESSOR(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # 移动到正确的设备
            if DEVICE is not None:
                target_device = DEVICE
            elif hasattr(OMNIVERIFIER_MODEL, "device"):
                target_device = OMNIVERIFIER_MODEL.device
            else:
                target_device = next(OMNIVERIFIER_MODEL.parameters()).device
            
            # 确保所有输入都在同一个设备上
            inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v 
                      for k, v in inputs.items()}
            
            # 推理（增加max_new_tokens以容纳think标签内容）
            with torch.no_grad():
                generated_ids = OMNIVERIFIER_MODEL.generate(**inputs, max_new_tokens=512,do_sample=False)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
                ]
                output_text = OMNIVERIFIER_PROCESSOR.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            
            # 提取answer
            response_text = output_text[0] if output_text else ""
            answer, explanation = extract_answer_from_response(response_text)
            
            # 如果答案为true，计数+1
            if answer:
                correct_count += 1
        
        # 计算得分：正确答案数 / 总问题数
        score = correct_count / total_questions if total_questions > 0 else 0.0
        
        return score
        
    except Exception as e:
        print(f"Error computing OmniVerifier score with QA: {e}", flush=True)
        raise


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查端点"""
    return jsonify({
        "status": "healthy",
        "model_loaded": OMNIVERIFIER_MODEL is not None,
        "processor_loaded": OMNIVERIFIER_PROCESSOR is not None,
    })


@app.route("/compute_reward", methods=["POST"])
def compute_reward_endpoint():
    """
    计算奖励API端点
    
    请求格式（JSON）:
    {
        "image": "base64编码的图像",
        "prompt": "文本提示",
        "reward_type": "omniverifier",  # 当前只支持omniverifier
        "generated_qa": {  # 可选，如果提供则使用多问题模式
            "yn_question_list": ["Is there a cup in the image?", "Is the cup red in color?"]
        }
    }
    
    返回格式（JSON）:
    {
        "success": true,
        "score": 1.0,  # 如果使用generated_qa，则为正确答案数/总问题数；否则1.0表示true，0.0表示false
        "raw_score": 1.0,
        "reward_type": "omniverifier",
        "error": null
    }
    """
    # 记录请求信息
    print(f"[REQUEST] POST /compute_reward | from: {request.remote_addr}", flush=True)
    try:
        data = request.get_json()
        
        # 解析输入
        image_b64 = data.get("image")
        prompt = data.get("prompt")
        reward_type = data.get("reward_type", "omniverifier")
        generated_qa = data.get("generated_qa")  # 新增：支持generated_qa字段
        
        # 确保generated_qa是dict类型（如果是从JSON反序列化的）
        if generated_qa is not None and not isinstance(generated_qa, dict):
            # 如果是字符串，尝试解析为JSON
            if isinstance(generated_qa, str):
                try:
                    generated_qa = json.loads(generated_qa)
                except json.JSONDecodeError:
                    generated_qa = None
            else:
                generated_qa = None
        
        # 记录请求参数（截断prompt避免日志过长）
        prompt_preview = prompt[:50] + "..." if prompt and len(prompt) > 50 else prompt
        has_qa = generated_qa is not None and isinstance(generated_qa, dict) and "yn_question_list" in generated_qa
        qa_count = len(generated_qa.get("yn_question_list", [])) if has_qa else 0
        print(f"[REQUEST] reward_type={reward_type}, prompt_preview={prompt_preview}, has_qa={has_qa}, qa_count={qa_count}", flush=True)
        
        if not image_b64:
            return jsonify({
                "success": False,
                "score": 0.0,
                "error": "Missing required field: image"
            }), 400
        
        # 如果使用generated_qa模式，prompt不是必需的
        if not has_qa and not prompt:
            return jsonify({
                "success": False,
                "score": 0.0,
                "error": "Missing required field: prompt (or generated_qa)"
            }), 400
        
        if reward_type != "omniverifier":
            return jsonify({
                "success": False,
                "score": 0.0,
                "error": f"Unsupported reward type: {reward_type}. Only 'omniverifier' is supported."
            }), 400
        
        # 解码图像
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 计算OmniVerifier分数
        if has_qa:
            # 使用多问题模式
            yn_question_list = generated_qa.get("yn_question_list", [])
            # 确保yn_question_list是list类型
            if not isinstance(yn_question_list, list):
                return jsonify({
                    "success": False,
                    "score": 0.0,
                    "error": f"generated_qa.yn_question_list must be a list, got {type(yn_question_list)}"
                }), 400
            if not yn_question_list:
                return jsonify({
                    "success": False,
                    "score": 0.0,
                    "error": "generated_qa.yn_question_list is empty"
                }), 400
            # 确保所有问题都是字符串
            yn_question_list = [str(q) for q in yn_question_list if q]
            if not yn_question_list:
                return jsonify({
                    "success": False,
                    "score": 0.0,
                    "error": "generated_qa.yn_question_list contains no valid questions"
                }), 400
            score = compute_omniverifier_score_with_qa(image, yn_question_list)
        else:
            # 使用原始单prompt模式
            score = compute_omniverifier_score(image, prompt)
        
        # 记录响应信息
        print(f"[RESPONSE] Success | score={score:.4f}, mode={'qa' if has_qa else 'prompt'}", flush=True)
        
        return jsonify({
            "success": True,
            "score": score,
            "raw_score": score,
            "reward_type": "omniverifier",
            "error": None
        })
        
    except Exception as e:
        print(f"[RESPONSE] Failed | error={str(e)[:100]}", flush=True)
        return jsonify({
            "success": False,
            "score": 0.0,
            "error": str(e)
        }), 500


def parse_args():
    parser = argparse.ArgumentParser(description="OmniVerifier奖励服务")
    parser.add_argument("--model_path", type=str, required=True, help="OmniVerifier模型路径")
    parser.add_argument("--port", type=int, default=5002, help="服务端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务host")
    parser.add_argument("--device", type=int, default=0, help="GPU设备ID")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 如果设置了 CUDA_VISIBLE_DEVICES，PyTorch 只能看到索引为 0 的 GPU
    # 因此需要使用 cuda:0 而不是原始的 GPU ID
    if torch.cuda.is_available():
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            # CUDA_VISIBLE_DEVICES 已设置，使用 cuda:0
            device = torch.device("cuda:0")
        else:
            # 未设置 CUDA_VISIBLE_DEVICES，使用指定的设备 ID
            device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    
    global DEVICE
    DEVICE = device
    
    # 加载模型
    load_omniverifier_model(args.model_path, device)
    
    # 启动服务
    print(f"🚀 Starting OmniVerifier reward server on {args.host}:{args.port}")
    print(f"   Model path: {args.model_path}")
    print(f"   Device: {device}")
    
    # 配置Flask日志：记录HTTP请求
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.INFO)  # 记录INFO级别（HTTP请求）
    # 确保werkzeug日志输出到标准输出（而不是stderr）
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.disabled = False
    
    app.run(host=args.host, port=args.port, threaded=True, processes=1)


if __name__ == "__main__":
    main()

