#!/usr/bin/env python3
"""
Qwen3-VL奖励服务 - 使用Qwen3-VL模型进行奖励计算
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

# 导入Qwen3-VL模型
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    QWEN3VL_AVAILABLE = True
except ImportError:
    QWEN3VL_AVAILABLE = False
    print("Warning: Qwen3-VL dependencies not available. Please install transformers")

app = Flask(__name__)

# 全局模型变量
QWEN3VL_MODEL = None
QWEN3VL_PROCESSOR = None
DEVICE = None


def load_qwen3vl_model(model_path: str, device: torch.device):
    """加载Qwen3-VL模型"""
    global QWEN3VL_MODEL, QWEN3VL_PROCESSOR
    
    if not QWEN3VL_AVAILABLE:
        raise RuntimeError("Qwen3-VL dependencies not available. Please install transformers")
    
    print(f"Loading Qwen3-VL model from {model_path}...")
    
    try:
        # 加载模型（使用bfloat16精度，使用device_map="auto"让transformers自动管理）
        # Qwen3-VL推荐使用auto类型和device_map="auto"
        QWEN3VL_MODEL = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # 使用自动设备映射
            trust_remote_code=True
        )
        
        # 如果想使用flash_attention_2加速（需要安装flash-attn）
        # QWEN3VL_MODEL = AutoModelForImageTextToText.from_pretrained(
        #     model_path,
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto",
        #     trust_remote_code=True
        # )
        
        # 加载processor
        QWEN3VL_PROCESSOR = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print(f"✅ Qwen3-VL model loaded successfully")
        print(f"   Model path: {model_path}")
        print(f"   Device map: auto")
        print(f"   Model dtype: {QWEN3VL_MODEL.dtype if hasattr(QWEN3VL_MODEL, 'dtype') else 'N/A'}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load Qwen3-VL model: {e}")


# def build_verification_prompt(original_prompt: str) -> str:
#     """构建验证prompt - 使用训练时的模板（omniverifier_with_edit.jinja）"""
#     question = f"""This image was generated from the prompt: {original_prompt}. Please carefully analyze the image and determine whether all the objects, attributes, and spatial relationships mentioned in the prompt are correctly represented in the image.

# If the image accurately reflects the prompt, please answer 'true'; otherwise, answer 'false'.

# When the answer is false, you must:
# 1. Identify the main error and describe it briefly in "explanation".
# 2. In "edit_prompt", provide a **concrete image editing instruction** to fix the error.
#    - Choose the most appropriate action based on the actual error: add / remove / replace / move / change color / modify attribute.
#    - The instruction must specify the exact action and the location or reference point (e.g., "replace the white candle on the right side with a white candle holder", "change the fork's color from silver to gold", "remove the pink toy vehicle attached to the airplane's nose").
#    - Do not give vague instructions such as "add more bottles" or "ensure the count is correct". Be precise and actionable.
#    - **Important**: Do not copy the template placeholder text. Write a real, specific instruction that addresses the actual error you identified.

# Examples of good edit_prompt instructions:
# - "Replace the white candle on the right side of the image with a white candle holder."
# - "Change the fork's color from silver to gold."
# - "Remove the pink toy vehicle attached to the airplane's nose and add a pink toaster next to the airplane."
# - "Change the orange's color from orange to green."

# Respond strictly in the following JSON format:

# {{
#     "answer": true/false,
#     "explanation": "A brief, specific description of the main error (if answer is false).",
#     "edit_prompt": "A concrete, location-specific editing instruction to fix the error (if answer is false)."
# }}

# You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"""
#     return question

# def build_verification_prompt(original_prompt: str) -> str:
#     """构建验证prompt"""
#     question = f"""This image was generated from the prompt: {original_prompt}. 
#     Please carefully analyze the image and determine whether all the objects, attributes, and spatial relationships mentioned in the prompt are correctly represented in the image. 

#     If the image accurately reflects the prompt, please answer 'true'; otherwise, answer 'false'.  

#     Respond strictly in the following JSON format: 

#     {{
#         "answer": true/false,
#         "explanation": "If the answer is false, briefly summarize the main error.",
#     }}
#     """
#     return question

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

    Respond strictly in the following JSON format:

    {{
        "answer": true/false,
        "explanation": "Brief explanation of your answer.",
    }}
    """
    return prompt


def extract_answer_from_response(response_text: str) -> tuple[bool, str]:
    """
    从模型响应中提取answer（增强容错，处理截断情况）
    
    Returns:
        (answer: bool, explanation: str)
    """
    # 优化：如果有 <think> 标签，先提取 </think> 之后的 JSON 部分
    if "</think>" in response_text:
        # 找到 </think> 后面的部分
        json_part = response_text.split("</think>", 1)[-1].strip()
    else:
        json_part = response_text.strip()
    
    try:
        # 尝试直接解析JSON
        response_json = json.loads(json_part)
        answer = response_json.get("answer", False)
        explanation = response_json.get("explanation", "")
        return bool(answer), explanation
    except json.JSONDecodeError as e:
        # JSON 可能被截断，尝试修复常见问题
        # 1. 缺少尾部的 }
        if not json_part.rstrip().endswith("}"):
            # 尝试补全缺失的引号和大括号
            json_part_fixed = json_part.rstrip()
            # 如果最后一个字段的值被截断（没有结束引号）
            if json_part_fixed.count('"') % 2 == 1:  # 奇数个引号，说明有一个引号没配对
                json_part_fixed += '"}'
            else:
                json_part_fixed += '}'
            
            try:
                response_json = json.loads(json_part_fixed)
                answer = response_json.get("answer", False)
                explanation = response_json.get("explanation", "")
                print(f"[WARNING] JSON was truncated, fix succeeded | original_error={str(e)[:50]}", flush=True)
                return bool(answer), explanation
            except json.JSONDecodeError:
                pass  # 修复失败，继续用正则
        
        # 2. 尝试找到最后一个完整的 } 前的部分
        last_brace = json_part.rfind("}")
        if last_brace > 0:
            json_part_truncated = json_part[:last_brace + 1]
            try:
                response_json = json.loads(json_part_truncated)
                answer = response_json.get("answer", False)
                explanation = response_json.get("explanation", "")
                print(f"[WARNING] Used truncated but valid JSON portion", flush=True)
                return bool(answer), explanation
            except json.JSONDecodeError:
                pass
        
        # 如果修复失败，使用正则表达式提取（至少能拿到 answer）
        answer_match = re.search(r'"answer"\s*:\s*(true|false)', response_text, re.IGNORECASE)
        if answer_match:
            answer_str = answer_match.group(1).lower()
            answer = answer_str == "true"
            
            # 尝试提取explanation
            explanation_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', response_text, re.IGNORECASE)
            explanation = explanation_match.group(1) if explanation_match else ""
            
            print(f"[WARNING] JSON parse failed, used regex fallback | error={str(e)[:80]}", flush=True)
            return answer, explanation
        
        # 如果都失败了，返回False
        print(f"[ERROR] Failed to parse response | error={str(e)}", flush=True)
        return False, "Failed to parse response"


def compute_qwen3vl_score(image: Image.Image, prompt: str) -> float:
    """
    使用Qwen3-VL模型计算奖励分数
    
    Args:
        image: PIL图像
        prompt: 原始prompt
    
    Returns:
        奖励分数：如果answer为true返回1.0，否则返回0.0
    """
    global QWEN3VL_MODEL, QWEN3VL_PROCESSOR
    
    if QWEN3VL_MODEL is None or QWEN3VL_PROCESSOR is None:
        raise RuntimeError("Qwen3-VL model not loaded")
    
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
        
        # 准备输入 - Qwen3-VL使用新的API
        inputs = QWEN3VL_PROCESSOR.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # 移动到模型设备
        inputs = inputs.to(QWEN3VL_MODEL.device)
        
        with torch.no_grad():
            generated_ids = QWEN3VL_MODEL.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # 监控生成长度，警告可能的截断
            gen_length = len(generated_ids_trimmed[0]) if generated_ids_trimmed else 0
            if gen_length >= 2045:  # 接近上限（留3个token余量）
                print(f"[WARNING] Generation may be truncated | length={gen_length}/2048", flush=True)
            
            output_text = QWEN3VL_PROCESSOR.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        # 提取answer
        response_text = output_text[0] if output_text else ""
        answer, explanation = extract_answer_from_response(response_text)
        
        # 返回分数：true=1.0, false=0.0
        score = 1.0 if answer else 0.0
        
        return score
        
    except Exception as e:
        print(f"Error computing Qwen3-VL score: {e}", flush=True)
        raise


def compute_qwen3vl_score_with_qa(image: Image.Image, yn_question_list: list[str]) -> float:
    """
    使用Qwen3-VL模型基于多个问题计算奖励分数
    
    Args:
        image: PIL图像
        yn_question_list: 是/否问题列表，例如 ["Is there a cup in the image?", "Is the cup red in color?"]
    
    Returns:
        奖励分数：正确答案数 / 总问题数，范围 [0.0, 1.0]
    """
    global QWEN3VL_MODEL, QWEN3VL_PROCESSOR
    
    if QWEN3VL_MODEL is None or QWEN3VL_PROCESSOR is None:
        raise RuntimeError("Qwen3-VL model not loaded")
    
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
            
            # 准备输入 - Qwen3-VL使用新的API
            inputs = QWEN3VL_PROCESSOR.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # 移动到模型设备
            inputs = inputs.to(QWEN3VL_MODEL.device)
            
            # 推理（使用 max_new_tokens=2048 以提供更大安全边际）
            with torch.no_grad():
                generated_ids = QWEN3VL_MODEL.generate(**inputs, max_new_tokens=2048)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                # 监控生成长度
                gen_length = len(generated_ids_trimmed[0]) if generated_ids_trimmed else 0
                if gen_length >= 2045:
                    print(f"[WARNING] QA generation may be truncated | length={gen_length}/2048", flush=True)
                
                output_text = QWEN3VL_PROCESSOR.batch_decode(
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
        print(f"Error computing Qwen3-VL score with QA: {e}", flush=True)
        raise


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查端点"""
    return jsonify({
        "status": "healthy",
        "model_loaded": QWEN3VL_MODEL is not None,
        "processor_loaded": QWEN3VL_PROCESSOR is not None,
    })


@app.route("/compute_reward", methods=["POST"])
def compute_reward_endpoint():
    """
    计算奖励API端点
    
    请求格式（JSON）:
    {
        "image": "base64编码的图像",
        "prompt": "文本提示",
        "reward_type": "qwen3vl",  # 当前只支持qwen3vl（向后兼容omniverifier）
        "generated_qa": {  # 可选，如果提供则使用多问题模式
            "yn_question_list": ["Is there a cup in the image?", "Is the cup red in color?"]
        }
    }
    
    返回格式（JSON）:
    {
        "success": true,
        "score": 1.0,  # 如果使用generated_qa，则为正确答案数/总问题数；否则1.0表示true，0.0表示false
        "raw_score": 1.0,
        "reward_type": "qwen3vl",
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
        reward_type = data.get("reward_type", "qwen3vl")
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
        
        if reward_type not in ["qwen3vl", "omniverifier"]:  # 保持向后兼容
            return jsonify({
                "success": False,
                "score": 0.0,
                "error": f"Unsupported reward type: {reward_type}. Only 'qwen3vl' is supported."
            }), 400
        
        # 解码图像
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 计算Qwen3-VL分数
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
            score = compute_qwen3vl_score_with_qa(image, yn_question_list)
        else:
            # 使用原始单prompt模式
            score = compute_qwen3vl_score(image, prompt)
        
        # 记录响应信息
        print(f"[RESPONSE] Success | score={score:.4f}, mode={'qa' if has_qa else 'prompt'}", flush=True)
        
        return jsonify({
            "success": True,
            "score": score,
            "raw_score": score,
            "reward_type": "qwen3vl",
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
    parser = argparse.ArgumentParser(description="Qwen3-VL奖励服务")
    parser.add_argument("--model_path", type=str, required=True, help="Qwen3-VL模型路径")
    parser.add_argument("--port", type=int, default=5002, help="服务端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务host")
    parser.add_argument("--device", type=int, default=0, help="GPU设备ID（当使用device_map='auto'时会被忽略）")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Qwen3-VL使用device_map="auto"，会自动管理设备分配
    # 但我们仍然设置DEVICE以便兼容性
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
    load_qwen3vl_model(args.model_path, device)
    
    # 启动服务
    print(f"🚀 Starting Qwen3-VL reward server on {args.host}:{args.port}")
    print(f"   Model path: {args.model_path}")
    print(f"   Device: {device} (Note: model uses device_map='auto')")
    
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

