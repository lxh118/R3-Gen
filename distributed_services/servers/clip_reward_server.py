#!/usr/bin/env python3
"""
CLIP奖励服务 - 轻量级奖励计算
支持多GPU负载均衡
"""

import argparse
import io
import base64
import os
import sys
import logging

import torch
from flask import Flask, request, jsonify
from PIL import Image

# 支持自定义CLIP模块路径
CLIP_MODULE_PATH = None
CLIP_AVAILABLE = False
clip = None

# 尝试加载标准CLIP
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    pass

app = Flask(__name__)

# 全局模型变量
CLIP_MODEL = None
CLIP_PREPROCESS = None
DEVICE = None


def load_clip_module(clip_module_path: str = None):
    """加载自定义CLIP模块"""
    global clip, CLIP_AVAILABLE, CLIP_MODULE_PATH
    
    if clip_module_path:
        clip_module_path = os.path.abspath(clip_module_path)
        if not os.path.exists(clip_module_path):
            raise RuntimeError(f"CLIP module path does not exist: {clip_module_path}")
        
        # 如果提供的是clip.py文件，获取其所在目录
        if clip_module_path.endswith('.py'):
            clip_dir = os.path.dirname(clip_module_path)
        else:
            clip_dir = clip_module_path
        
        # 将CLIP模块的父目录添加到sys.path（这样可以通过import clip导入）
        clip_parent_dir = os.path.dirname(clip_dir)
        if clip_parent_dir not in sys.path:
            sys.path.insert(0, clip_parent_dir)
        
        # 导入自定义CLIP模块
        try:
            # 移除可能存在的旧clip模块
            if 'clip' in sys.modules:
                del sys.modules['clip']
            
            # 导入CLIP模块（从目录导入，因为目录下有__init__.py）
            import clip as custom_clip
            clip = custom_clip
            CLIP_AVAILABLE = True
            CLIP_MODULE_PATH = clip_module_path
            print(f"✅ Loaded custom CLIP module from: {clip_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to load custom CLIP module from {clip_module_path}: {e}")
    elif not CLIP_AVAILABLE:
        raise RuntimeError("CLIP is not available. Please install: pip install git+https://github.com/openai/CLIP.git or specify --clip_module_path")


def load_clip_model(model_name: str, device: torch.device, model_path: str = None):
    """加载CLIP模型"""
    global CLIP_MODEL, CLIP_PREPROCESS
    
    if not CLIP_AVAILABLE:
        raise RuntimeError("CLIP is not available. Please install: pip install git+https://github.com/openai/CLIP.git")
    
    # 如果提供了模型路径，优先使用
    if model_path and os.path.isfile(model_path):
        print(f"Loading CLIP model from checkpoint: {model_path}...")
        CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_path, device=device)
    else:
        print(f"Loading CLIP model: {model_name}...")
        
        # 在加载前，先检查并清理可能损坏的缓存文件
        # 注意：所有CLIP服务实例都在同一个节点上，共享缓存目录
        # 使用文件锁确保只有一个实例会删除缓存文件，避免影响其他实例
        cache_dir = os.path.expanduser("~/.cache/clip")
        lock_file = os.path.join(cache_dir, ".cache_lock")
        
        if os.path.exists(cache_dir):
            # 检查可能的模型文件名
            model_name_clean = model_name.replace('/', '-').replace('_', '-')
            possible_files = [
                os.path.join(cache_dir, f"{model_name_clean}.pt"),
                os.path.join(cache_dir, f"{model_name.replace('/', '_')}.pt"),
                os.path.join(cache_dir, f"{model_name}.pt"),
            ]
            # 添加常见的CLIP模型名称映射
            if "ViT-B/32" in model_name:
                possible_files.append(os.path.join(cache_dir, "ViT-B-32.pt"))
            if "RN50" in model_name:
                possible_files.append(os.path.join(cache_dir, "RN50.pt"))
            
            # 如果缓存文件存在，CLIP库会自动检测并重新下载（如果校验失败）
            # 但为了保险，我们可以先检查文件大小是否异常小（可能是下载中断）
            # 使用文件锁确保只有一个实例会删除缓存文件
            import fcntl
            try:
                # 创建锁文件（如果不存在）
                os.makedirs(cache_dir, exist_ok=True)
                with open(lock_file, 'w') as lock:
                    try:
                        # 尝试获取非阻塞锁（如果其他实例正在操作，跳过）
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        
                        for cache_file in possible_files:
                            if os.path.exists(cache_file):
                                file_size = os.path.getsize(cache_file)
                                # ViT-B/32模型应该是338MB左右，如果文件明显小于这个值，可能是下载不完整
                                if file_size < 300 * 1024 * 1024:  # 小于300MB
                                    print(f"⚠️  检测到可能不完整的缓存文件: {cache_file} (大小: {file_size / 1024 / 1024:.1f}MB)", flush=True)
                                    print(f"  删除不完整的缓存文件，将重新下载...", flush=True)
                                    try:
                                        os.remove(cache_file)
                                    except Exception as e:
                                        print(f"  警告: 删除缓存文件失败: {e}", flush=True)
                        
                        # 释放锁
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                    except BlockingIOError:
                        # 其他实例正在操作，跳过删除操作
                        print(f"  其他实例正在操作缓存，跳过预检查...", flush=True)
            except ImportError:
                # Windows系统不支持fcntl，跳过文件锁
                print(f"  警告: 不支持文件锁（非Linux系统），跳过预检查", flush=True)
            except Exception as e:
                print(f"  警告: 文件锁操作失败: {e}，跳过预检查", flush=True)
        
        # 在加载模型前，检查缓存文件是否存在且完整
        # 使用文件锁确保只有一个实例下载，其他实例等待
        cache_dir = os.path.expanduser("~/.cache/clip")
        lock_file = os.path.join(cache_dir, ".download_lock")
        model_cache_file = None
        
        # 确定模型缓存文件路径
        if "ViT-B/32" in model_name:
            model_cache_file = os.path.join(cache_dir, "ViT-B-32.pt")
        else:
            model_name_clean = model_name.replace('/', '-').replace('_', '-')
            model_cache_file = os.path.join(cache_dir, f"{model_name_clean}.pt")
        
        # 使用文件锁机制，确保只有一个实例下载
        import fcntl
        import time
        cache_exists = False
        is_downloader = False  # 标记当前实例是否是负责下载的实例
        
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(lock_file, 'w') as lock:
                # 先尝试非阻塞获取锁
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # 获取锁成功，当前实例负责检查和下载
                    is_downloader = True
                    
                    # 检查缓存文件是否存在且完整
                    if model_cache_file and os.path.exists(model_cache_file):
                        file_size = os.path.getsize(model_cache_file)
                        if file_size >= 300 * 1024 * 1024:  # 至少300MB
                            cache_exists = True
                            print(f"✅ 发现已存在的模型缓存: {model_cache_file} (大小: {file_size / 1024 / 1024:.1f}MB)", flush=True)
                            # 释放锁，让其他实例继续
                            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                        else:
                            print(f"📥 缓存文件不完整，当前实例负责下载...", flush=True)
                            # 保持锁，继续下载
                    else:
                        print(f"📥 缓存文件不存在，当前实例负责下载...", flush=True)
                        # 保持锁，继续下载
                        
                except BlockingIOError:
                    # 获取锁失败，说明其他实例正在下载，等待其完成
                    print(f"⏳ 其他实例正在下载模型，等待下载完成...", flush=True)
                    # 释放非阻塞锁尝试，改用阻塞锁等待
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)  # 阻塞等待锁释放
                    # 锁释放后，说明其他实例下载完成（或失败）
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                    
                    # 再次检查缓存文件
                    max_wait = 120  # 最多等待120秒
                    wait_interval = 2  # 每2秒检查一次
                    waited = 0
                    while waited < max_wait:
                        if model_cache_file and os.path.exists(model_cache_file):
                            file_size = os.path.getsize(model_cache_file)
                            if file_size >= 300 * 1024 * 1024:
                                print(f"✅ 其他实例已完成下载，使用现有缓存 (大小: {file_size / 1024 / 1024:.1f}MB)", flush=True)
                                cache_exists = True
                                break
                        time.sleep(wait_interval)
                        waited += wait_interval
                        if waited % 10 == 0:
                            print(f"   等待中... (已等待 {waited}秒 / {max_wait}秒)", flush=True)
                    
                    if not cache_exists:
                        print(f"⚠️  等待超时，将尝试重新下载...", flush=True)
                        is_downloader = True  # 如果等待超时，当前实例负责下载
                        
        except ImportError:
            # Windows系统不支持fcntl
            print(f"⚠️  不支持文件锁（非Linux系统），直接检查缓存", flush=True)
            if model_cache_file and os.path.exists(model_cache_file):
                file_size = os.path.getsize(model_cache_file)
                if file_size >= 300 * 1024 * 1024:
                    cache_exists = True
                    print(f"✅ 发现已存在的模型缓存: {model_cache_file} (大小: {file_size / 1024 / 1024:.1f}MB)", flush=True)
        except Exception as e:
            print(f"⚠️  文件锁操作失败: {e}，直接检查缓存", flush=True)
            if model_cache_file and os.path.exists(model_cache_file):
                file_size = os.path.getsize(model_cache_file)
                if file_size >= 300 * 1024 * 1024:
                    cache_exists = True
                    print(f"✅ 发现已存在的模型缓存: {model_cache_file} (大小: {file_size / 1024 / 1024:.1f}MB)", flush=True)
        
        # 如果缓存已存在，等待一小段时间确保文件完整
        if cache_exists:
            time.sleep(0.5)  # 等待0.5秒，确保文件写入完成
        
        # 添加错误处理：如果SHA256校验失败，清除缓存并重试
        # 增加重试次数，因为网络问题可能导致多次下载失败
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 如果缓存不存在或当前实例是下载者，在持有锁的情况下调用clip.load()
                if not cache_exists and is_downloader:
                    try:
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(lock_file, 'w') as lock:
                            # 获取锁（阻塞模式，确保只有一个实例下载）
                            print(f"🔒 获取下载锁（当前实例负责下载，其他实例将等待）...", flush=True)
                            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                            
                            # 再次检查缓存（可能其他实例已经下载完成）
                            if model_cache_file and os.path.exists(model_cache_file):
                                file_size = os.path.getsize(model_cache_file)
                                if file_size >= 300 * 1024 * 1024:
                                    print(f"✅ 其他实例已完成下载，使用现有缓存", flush=True)
                                    cache_exists = True
                                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                                else:
                                    print(f"📥 开始下载CLIP模型（当前实例负责下载，其他实例将等待）...", flush=True)
                                    # 持有锁的情况下调用clip.load()，确保只有一个实例下载
                                    # 注意：锁会在with块结束时自动释放
                                    CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                                    print(f"✅ 模型下载完成，释放锁", flush=True)
                                    # 释放锁，让其他实例继续
                                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                                    break
                            else:
                                print(f"📥 开始下载CLIP模型（当前实例负责下载，其他实例将等待）...", flush=True)
                                # 持有锁的情况下调用clip.load()，确保只有一个实例下载
                                CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                                print(f"✅ 模型下载完成，释放锁", flush=True)
                                # 释放锁，让其他实例继续
                                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                                break
                    except ImportError:
                        # Windows系统不支持fcntl，先检查缓存
                        print(f"⚠️  不支持文件锁（非Linux系统），检查缓存...", flush=True)
                        if model_cache_file and os.path.exists(model_cache_file):
                            file_size = os.path.getsize(model_cache_file)
                            if file_size >= 300 * 1024 * 1024:
                                print(f"✅ 发现已存在的模型缓存，使用缓存", flush=True)
                                cache_exists = True
                            else:
                                print(f"⚠️  缓存文件不完整，将重新下载", flush=True)
                                cache_exists = False
                        if not cache_exists:
                            print(f"📥 开始下载CLIP模型...", flush=True)
                            CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                        else:
                            CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                        break
                    except Exception as e:
                        # 文件锁操作失败，先检查缓存是否存在
                        print(f"⚠️  文件锁操作失败: {e}，检查缓存...", flush=True)
                        # 等待一小段时间，让其他实例完成下载
                        import time
                        time.sleep(2)
                        # 检查缓存是否存在且完整
                        if model_cache_file and os.path.exists(model_cache_file):
                            file_size = os.path.getsize(model_cache_file)
                            if file_size >= 300 * 1024 * 1024:
                                print(f"✅ 发现已存在的模型缓存，使用缓存 (大小: {file_size / 1024 / 1024:.1f}MB)", flush=True)
                                cache_exists = True
                                CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                                break
                        # 如果缓存不存在，等待更长时间后重试获取锁
                        print(f"⏳ 缓存不存在，等待其他实例下载（最多30秒）...", flush=True)
                        max_wait_retry = 30
                        wait_interval_retry = 2
                        waited_retry = 0
                        while waited_retry < max_wait_retry:
                            if model_cache_file and os.path.exists(model_cache_file):
                                file_size = os.path.getsize(model_cache_file)
                                if file_size >= 300 * 1024 * 1024:
                                    print(f"✅ 其他实例已完成下载，使用现有缓存 (大小: {file_size / 1024 / 1024:.1f}MB)", flush=True)
                                    cache_exists = True
                                    CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                                    break
                            time.sleep(wait_interval_retry)
                            waited_retry += wait_interval_retry
                        if cache_exists:
                            break
                        # 如果等待超时，尝试重新获取锁并下载
                        print(f"⚠️  等待超时，尝试重新获取锁并下载...", flush=True)
                        try:
                            os.makedirs(cache_dir, exist_ok=True)
                            with open(lock_file, 'w') as lock_retry:
                                fcntl.flock(lock_retry.fileno(), fcntl.LOCK_EX)
                                # 再次检查缓存
                                if model_cache_file and os.path.exists(model_cache_file):
                                    file_size = os.path.getsize(model_cache_file)
                                    if file_size >= 300 * 1024 * 1024:
                                        print(f"✅ 其他实例已完成下载，使用现有缓存", flush=True)
                                        cache_exists = True
                                        fcntl.flock(lock_retry.fileno(), fcntl.LOCK_UN)
                                        CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                                        break
                                # 如果缓存仍不存在，当前实例负责下载
                                print(f"📥 开始下载CLIP模型（当前实例负责下载）...", flush=True)
                                CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                                fcntl.flock(lock_retry.fileno(), fcntl.LOCK_UN)
                                break
                        except Exception as retry_error:
                            # 如果重试仍然失败，作为最后手段直接下载（但会记录警告）
                            print(f"⚠️  重试获取锁失败: {retry_error}，作为最后手段直接下载（可能导致并发问题）", flush=True)
                            CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                            break
                elif not cache_exists and not is_downloader:
                    # 不是下载者，但缓存不存在，等待其他实例下载
                    print(f"⏳ 等待其他实例下载模型...", flush=True)
                    max_wait = 180  # 增加等待时间到180秒
                    wait_interval = 2
                    waited = 0
                    while waited < max_wait:
                        # 尝试获取锁（非阻塞），如果获取成功说明下载已完成
                        try:
                            with open(lock_file, 'w') as lock:
                                try:
                                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                                    # 获取锁成功，说明下载已完成，检查缓存
                                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                                    if model_cache_file and os.path.exists(model_cache_file):
                                        file_size = os.path.getsize(model_cache_file)
                                        if file_size >= 300 * 1024 * 1024:
                                            print(f"✅ 其他实例已完成下载，使用现有缓存 (大小: {file_size / 1024 / 1024:.1f}MB)", flush=True)
                                            cache_exists = True
                                            break
                                except BlockingIOError:
                                    # 锁被占用，说明正在下载，继续等待
                                    pass
                        except Exception:
                            pass
                        
                        time.sleep(wait_interval)
                        waited += wait_interval
                        if waited % 10 == 0:
                            print(f"   等待中... (已等待 {waited}秒 / {max_wait}秒)", flush=True)
                    
                    if cache_exists:
                        CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                        break
                    else:
                        # 等待超时，当前实例尝试下载
                        print(f"⚠️  等待超时，当前实例尝试下载...", flush=True)
                        is_downloader = True
                        cache_exists = False
                        continue
                else:
                    # 缓存已存在，直接加载
                    CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                    break
            except (RuntimeError, Exception) as e:
                error_msg = str(e).lower()
                # 检查是否是SHA256校验失败或其他下载相关错误
                is_checksum_error = (
                    "sha256 checksum" in error_msg or 
                    "checksum" in error_msg or
                    "does not not match" in error_msg or
                    "does not match" in error_msg
                )
                
                if is_checksum_error:
                    if attempt < max_retries - 1:
                        print(f"⚠️  CLIP模型校验失败，清除缓存并重试 (尝试 {attempt + 1}/{max_retries})...", flush=True)
                        print(f"  错误信息: {e}", flush=True)
                        
                        # 清除CLIP模型缓存（更彻底）
                        # 注意：使用文件锁确保只有一个实例会删除缓存，避免影响其他实例
                        cache_dir = os.path.expanduser("~/.cache/clip")
                        lock_file = os.path.join(cache_dir, ".download_lock")  # 使用统一的锁文件
                        # 删除缓存后，更新cache_exists状态
                        cache_exists = False
                        is_downloader = True  # 校验失败后，当前实例负责重新下载
                        
                        if os.path.exists(cache_dir):
                            import shutil
                            import fcntl
                            try:
                                # 创建锁文件（如果不存在）
                                os.makedirs(cache_dir, exist_ok=True)
                                with open(lock_file, 'w') as lock:
                                    try:
                                        # 尝试获取非阻塞锁
                                        # 如果其他实例正在操作，等待一段时间后重试
                                        max_lock_wait = 10  # 最多等待10秒
                                        lock_acquired = False
                                        for lock_attempt in range(max_lock_wait):
                                            try:
                                                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                                                lock_acquired = True
                                                break
                                            except BlockingIOError:
                                                if lock_attempt < max_lock_wait - 1:
                                                    import time
                                                    time.sleep(1)
                                                    continue
                                                else:
                                                    print(f"  警告: 无法获取文件锁，其他实例可能正在操作缓存", flush=True)
                                                    print(f"  跳过删除操作，直接重试下载...", flush=True)
                                        
                                        if lock_acquired:
                                            # CLIP库的模型文件名格式：ViT-B-32.pt, RN50.pt 等
                                            # 尝试匹配可能的模型文件名
                                            model_name_clean = model_name.replace('/', '-').replace('_', '-')
                                            possible_patterns = [
                                                f"{model_name_clean}.pt",
                                                f"{model_name.replace('/', '_')}.pt",
                                                f"{model_name}.pt",
                                            ]
                                            # 添加常见的CLIP模型名称映射
                                            if "ViT-B/32" in model_name:
                                                possible_patterns.append("ViT-B-32.pt")
                                            if "RN50" in model_name:
                                                possible_patterns.append("RN50.pt")
                                            
                                            # 删除匹配的缓存文件
                                            deleted_any = False
                                            for pattern in possible_patterns:
                                                cache_file = os.path.join(cache_dir, pattern)
                                                if os.path.exists(cache_file):
                                                    print(f"  删除损坏的缓存文件: {cache_file}", flush=True)
                                                    try:
                                                        os.remove(cache_file)
                                                        deleted_any = True
                                                    except Exception as remove_error:
                                                        print(f"  警告: 删除 {cache_file} 失败: {remove_error}", flush=True)
                                            
                                            # 同时删除可能的临时文件（CLIP库可能在下载时创建临时文件）
                                            temp_patterns = [
                                                f"{model_name_clean}.pt.tmp",
                                                f"{model_name.replace('/', '_')}.pt.tmp",
                                                "ViT-B-32.pt.tmp" if "ViT-B/32" in model_name else None,
                                            ]
                                            for pattern in temp_patterns:
                                                if pattern is None:
                                                    continue
                                                temp_file = os.path.join(cache_dir, pattern)
                                                if os.path.exists(temp_file):
                                                    try:
                                                        os.remove(temp_file)
                                                        print(f"  删除临时文件: {temp_file}", flush=True)
                                                    except Exception:
                                                        pass
                                            
                                            # 如果没有找到特定文件，尝试删除整个缓存目录（更彻底）
                                            # 但要注意：这会影响到其他实例，所以只在没有找到任何匹配文件时才这样做
                                            if not deleted_any:
                                                print(f"  未找到特定模型文件，但不清除整个缓存目录（避免影响其他实例）", flush=True)
                                            else:
                                                print(f"  已清除损坏的模型缓存文件", flush=True)
                                            
                                            # 释放锁
                                            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                                    except ImportError:
                                        # Windows系统不支持fcntl，直接删除（单实例场景）
                                        print(f"  警告: 不支持文件锁（非Linux系统），直接删除缓存", flush=True)
                                        # 执行删除逻辑（简化版，不删除整个目录）
                                        model_name_clean = model_name.replace('/', '-').replace('_', '-')
                                        if "ViT-B/32" in model_name:
                                            cache_file = os.path.join(cache_dir, "ViT-B-32.pt")
                                            if os.path.exists(cache_file):
                                                try:
                                                    os.remove(cache_file)
                                                    print(f"  删除损坏的缓存文件: {cache_file}", flush=True)
                                                except Exception:
                                                    pass
                            except Exception as cleanup_error:
                                print(f"  警告: 清除缓存时出错: {cleanup_error}", flush=True)
                                import traceback
                                traceback.print_exc()
                        
                        # 等待更长时间再重试，避免网络拥塞和立即重试
                        # 每次重试等待时间递增：1秒、3秒、5秒
                        import time
                        wait_time = 1 + (attempt * 2)
                        print(f"  等待 {wait_time} 秒后重试...", flush=True)
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"❌ CLIP模型加载失败（已重试 {max_retries} 次）", flush=True)
                        print(f"  最后错误: {e}", flush=True)
                        print(f"  建议: 手动清除缓存目录 ~/.cache/clip 并重新启动服务", flush=True)
                        raise
                else:
                    # 其他类型的错误，直接抛出
                    print(f"❌ CLIP模型加载失败: {e}", flush=True)
                    raise
    
    print(f"✅ CLIP model loaded successfully on {device}")


@torch.no_grad()
def compute_clip_score(image: Image.Image, prompt: str) -> float:
    """计算CLIP余弦相似度（原始值，范围约 [-1, 1]）"""
    try:
        image_tensor = CLIP_PREPROCESS(image).unsqueeze(0).to(DEVICE)
        text_tokens = clip.tokenize([prompt]).to(DEVICE)
        
        image_features = CLIP_MODEL.encode_image(image_tensor)
        text_features = CLIP_MODEL.encode_text(text_tokens)
        
        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 计算相似度
        similarity = (image_features @ text_features.T).item()
        # 返回原始余弦相似度，不做区间映射
        return similarity
    except Exception as e:
        print(f"CLIP score computation error: {e}")
        return 0.0


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查"""
    return jsonify({"status": "healthy", "service": "clip_reward"})


@app.route("/compute_reward", methods=["POST"])
def compute_reward_endpoint():
    """
    计算奖励API端点
    
    请求格式（JSON）:
    {
        "image": "base64编码的图像",
        "prompt": "文本提示",
        "reward_type": "clip"  # 当前只支持clip
    }
    
    返回格式（JSON）:
    {
        "success": true,
        "score": 0.85,
        "raw_score": 0.85,
        "reward_type": "clip",
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
        reward_type = data.get("reward_type", "clip")
        
        if not image_b64 or not prompt:
            return jsonify({
                "success": False,
                "score": 0.0,
                "error": "Missing required fields: image or prompt"
            }), 400
        
        if reward_type != "clip":
            return jsonify({
                "success": False,
                "score": 0.0,
                "error": f"Unsupported reward type: {reward_type}. Only 'clip' is supported."
            }), 400
        
        # 解码图像
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 记录请求参数（截断prompt避免日志过长）
        prompt_preview = prompt[:50] + "..." if prompt and len(prompt) > 50 else prompt
        print(f"[REQUEST] reward_type={reward_type}, prompt_preview={prompt_preview}", flush=True)
        
        # 计算CLIP分数（原始余弦相似度）
        score = compute_clip_score(image, prompt)
        
        # 记录响应信息
        print(f"[RESPONSE] Success | score={score:.4f}", flush=True)
        
        return jsonify({
            "success": True,
            "score": score,      # 直接返回原始余弦相似度
            "raw_score": score,  # 保持字段一致
            "reward_type": "clip",
            "error": None
        })
        
    except Exception as e:
        print(f"[RESPONSE] Failed | error={str(e)[:100]}", flush=True)
        return jsonify({
            "success": False,
            "score": 0.0,
            "error": str(e)
        }), 500


@app.route("/compute_reward_batch", methods=["POST"])
def compute_reward_batch_endpoint():
    """
    批量计算奖励API端点
    
    请求格式（JSON）:
    {
        "batch": [
            {
                "image": "base64编码的图像",
                "prompt": "文本提示",
                "reward_type": "clip"
            },
            ...
        ]
    }
    
    返回格式（JSON）:
    {
        "success": true,
        "results": [
            {
                "score": 0.85,
                "raw_score": 0.85,
                "reward_type": "clip"
            },
            ...
        ],
        "error": null
    }
    """
    try:
        data = request.get_json()
        batch = data.get("batch", [])
        
        if not batch:
            return jsonify({
                "success": False,
                "results": [],
                "error": "Empty batch"
            }), 400
        
        results = []
        for item in batch:
            try:
                image_b64 = item.get("image")
                prompt = item.get("prompt")
                reward_type = item.get("reward_type", "clip")
                
                if not image_b64 or not prompt:
                    results.append({
                        "score": 0.0,
                        "raw_score": 0.0,
                        "reward_type": reward_type,
                        "error": "Missing image or prompt"
                    })
                    continue
                
                # 解码图像
                image_bytes = base64.b64decode(image_b64)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                # 计算分数（原始余弦相似度）
                score = compute_clip_score(image, prompt)
                
                results.append({
                    "score": score,
                    "raw_score": score,
                    "reward_type": "clip",
                    "error": None
                })
            except Exception as e:
                results.append({
                    "score": 0.0,
                    "raw_score": 0.0,
                    "reward_type": "clip",
                    "error": str(e)
                })
        
        return jsonify({
            "success": True,
            "results": results,
            "error": None
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "results": [],
            "error": str(e)
        }), 500


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP奖励服务")
    parser.add_argument("--model_name", type=str, default="ViT-B/32", help="CLIP模型名称（如果使用标准CLIP）")
    parser.add_argument("--model_path", type=str, default=None, help="CLIP模型checkpoint路径（.pt文件）")
    parser.add_argument("--clip_module_path", type=str, default=None, help="自定义CLIP模块路径（clip.py文件路径）")
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
    
    # 加载自定义CLIP模块（如果指定）
    if args.clip_module_path:
        load_clip_module(args.clip_module_path)
    
    # 加载模型
    load_clip_model(args.model_name, device, args.model_path)
    
    # 启动服务
    print(f"🚀 Starting CLIP reward server on {args.host}:{args.port}")
    if args.model_path:
        print(f"   Model checkpoint: {args.model_path}")
    else:
        print(f"   Model: {args.model_name}")
    if args.clip_module_path:
        print(f"   CLIP module: {args.clip_module_path}")
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

