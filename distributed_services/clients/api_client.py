"""
API客户端 - 用于调用图像编辑和奖励计算服务

放置在: reward-server/distributed_services/clients/api_client.py
"""

import base64
import io
import os
import random
import threading
import time
from typing import List, Optional, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image


def _get_env_int(key: str, default: int) -> int:
    """

    Args:
    key: 环境变量键
    default: 默认值

    Returns:
    整数值
    """
    value = os.environ.get(key, str(default))
    value = value.strip().strip('"').strip("'")
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _get_env_bool(key: str, default: bool = True) -> bool:
    """

    Args:
    key: 环境变量键
    default: 默认值

    Returns:
    布尔值
    """
    value = os.environ.get(key, str(default).lower())
    value = value.strip().strip('"').strip("'").lower()
    return value == "true"


class ServiceClient:
    """服务客户端基类（简化健康检查与重试逻辑）"""

    def __init__(
        self,
        endpoints: List[str],
        timeout: int = 180,
        max_retries: int = 3,
        health_check_timeout: int = 5,
        health_check_interval: int = 600,
        enable_health_check: bool = True
        ):
        """
        Args:
            endpoints: 服务端点列表（用于负载均衡）
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            health_check_timeout: 健康检查超时时间（秒）
            health_check_interval: 健康检查间隔（秒），定期检查所有端点
            enable_health_check: 是否启用健康检查（仅影响失败标记和打印）
        """
        self.endpoints = endpoints
        self.timeout = timeout
        self.max_retries = max_retries
        self.health_check_timeout = health_check_timeout
        self.health_check_interval = health_check_interval
        self.enable_health_check = enable_health_check
        # Round-robin 指针：
        # - 需要线程安全（训练端会用 ThreadPoolExecutor 并发请求）
        # - 需要多进程更均匀（多个 worker 进程都从0开始会导致热点）
        self._rr_lock = threading.Lock()
        self._current_index = random.randint(0, max(0, len(endpoints) - 1))

        # 健康状态跟踪
        # endpoint -> (is_healthy: bool, last_check_time: float, consecutive_failures: int)
        self._health_status: Dict[str, tuple] = {}
        self._last_full_health_check: float = 0.0
        self._last_request_time: float = 0.0  # 最后一次请求时间（用于判断是否需要定期检查）

        # 健康检查日志控制：记录上次健康检查的结果，只在状态变化时打印
        self._last_health_summary: Optional[tuple[int, int]] = None  # (healthy_count, unhealthy_count)

        # 全量健康检查日志控制：避免频繁输出"已尝试所有端点"的日志
        self._last_full_check_log_time: float = 0.0  # 上次打印全量检查日志的时间
        self._full_check_log_interval: float = 60.0  # 全量检查日志的最小间隔（秒）

        # 不健康端点日志控制：减少频繁失败端点的日志输出
        self._unhealthy_endpoint_log_count: Dict[str, int] = {}  # endpoint -> 已打印日志次数
        self._unhealthy_log_interval: int = 50  # 每N次失败才打印一次日志（当失败次数 > 50时）

        # 端点更新锁（用于线程安全）
        self._endpoints_lock = False

        # 初始化所有端点为健康状态（首次使用前会检查）
        for endpoint in endpoints:
            self._health_status[endpoint] = (True, 0.0, 0)

    def update_endpoints(self, new_endpoints: List[str]) -> bool:
        """更新端点列表（线程安全）"""
        if set(new_endpoints) == set(self.endpoints):
            return False

        # 简单的锁机制（避免并发更新）
        if self._endpoints_lock:
            return False

        self._endpoints_lock = True
        try:
            old_count = len(self.endpoints)
            self.endpoints = new_endpoints
            self._current_index = 0

            # 更新健康状态（保留已存在的，添加新的）
            new_health_status = {}
            for endpoint in new_endpoints:
                if endpoint in self._health_status:
                    new_health_status[endpoint] = self._health_status[endpoint]
                else:
                    new_health_status[endpoint] = (True, 0.0, 0)
            self._health_status = new_health_status

            # 更新模型类型缓存（保留已存在的，移除不存在的）
            if hasattr(self, '_model_types'):
                new_model_types = {}
                for endpoint in new_endpoints:
                    if endpoint in self._model_types:
                        new_model_types[endpoint] = self._model_types[endpoint]
                self._model_types = new_model_types

            print(f"[INFO] 端点已更新: {old_count} -> {len(new_endpoints)} 个端点")
            return True
        finally:
            self._endpoints_lock = False

    def _get_healthy_endpoints(self, force_check: bool = False) -> List[str]:
        """
        获取健康的端点列表

        Args:
        force_check: 是否强制检查所有端点

        Returns:
        健康的端点列表
        """
        if not self.enable_health_check:
            return self.endpoints

        current_time = time.time()

        # 被动健康检查为主，主动检查为辅
        # 只在以下情况才进行主动全量检查：
        # 1. 强制检查（所有端点都不健康时）
        # 2. 长时间没有请求时（600秒，远大于业务请求间隔）
        # 3. 首次使用（初始化时）
        long_idle_time = 600  # 600秒没有请求才进行主动检查
        should_full_check = (
            force_check or
            (self._last_request_time > 0 and (current_time - self._last_request_time) >= long_idle_time) or
            (self._last_full_health_check == 0.0)  # 首次使用
        )

        if should_full_check:
            # 添加随机延迟，避免多个客户端同时进行健康检查
            # VERL可能有多个worker进程，每个都有自己的客户端实例
            # 如果所有客户端同时检查，会导致某个端点收到大量并发请求
            # 延迟范围进一步减少（0-1秒），降低开销
            if not force_check:
                # 随机延迟0-1秒，分散健康检查请求（进一步减少延迟时间）
                random_delay = random.uniform(0, 1)
                time.sleep(random_delay)

            # 并发健康检查，而不是串行检查
            # 串行检查32个端点需要32×0.1秒=3.2秒，并发检查可以大大减少时间
            # 使用线程池并发检查所有端点，最多16个并发（避免对服务端造成过大压力）
            healthy_endpoints = []
            unhealthy_endpoints = []

            def check_single_endpoint(endpoint: str) -> tuple[str, bool]:
                """检查单个端点并返回结果"""
                is_healthy = self._check_endpoint_health(endpoint)
                return endpoint, is_healthy

            # 并发检查所有端点（最多16个并发，避免对服务端造成过大压力）
            max_concurrent_checks = min(16, len(self.endpoints))
            with ThreadPoolExecutor(max_workers=max_concurrent_checks) as executor:
                future_to_endpoint = {
                    executor.submit(check_single_endpoint, endpoint): endpoint
                    for endpoint in self.endpoints
                }

                for future in as_completed(future_to_endpoint):
                    try:
                        endpoint, is_healthy = future.result()
                        if is_healthy:
                            healthy_endpoints.append(endpoint)
                        else:
                            unhealthy_endpoints.append(endpoint)
                    except Exception as e:
                        # 如果检查失败，认为不健康
                        endpoint = future_to_endpoint[future]
                        unhealthy_endpoints.append(endpoint)
                        print(f"[WARN] 健康检查异常: {endpoint}, 错误: {e}")

            self._last_full_health_check = current_time

            current_summary = (len(healthy_endpoints), len(unhealthy_endpoints))

            if healthy_endpoints:
                # 日志输出策略：
                # 1. 首次检查且全部健康：不打印（避免初始化时的噪音）
                # 2. 首次检查且有部分不健康：打印（重要信息）
                # 3. 状态变化（从不健康变为健康，或健康数量显著变化）：打印
                # 4. 强制检查且全部健康：不打印（避免频繁输出）
                should_log = False
                if self._last_health_summary is None:
                    # 第一次检查：只在有不健康端点时打印
                    if len(unhealthy_endpoints) > 0:
                        should_log = True
                elif current_summary != self._last_health_summary:
                    # 状态发生变化，需要判断是否值得打印
                    last_unhealthy = self._last_health_summary[1]

                    if len(unhealthy_endpoints) > 0:
                        # 有不健康端点时，只在数量显著变化时打印（变化超过2个）
                        if abs(len(unhealthy_endpoints) - last_unhealthy) >= 2:
                            should_log = True
                    elif last_unhealthy > 0:
                        # 从有不健康变为全部健康，打印恢复信息（重要）
                        should_log = True
                    elif self._last_health_summary[1] > 0 and len(unhealthy_endpoints) == 0:
                        # 从有不健康变为全部健康，打印恢复信息
                        should_log = True

                if should_log:
                    if len(unhealthy_endpoints) > 0:
                        print(
                        f"[INFO] 健康检查结果: {len(healthy_endpoints)}/{len(self.endpoints)} 个端点健康，不健康端点: {len(unhealthy_endpoints)}")
                    else:
                        print(f"[INFO] 健康检查结果: {len(healthy_endpoints)}/{len(self.endpoints)} 个端点健康")

                self._last_health_summary = current_summary
                return healthy_endpoints
            else:
                # 所有端点都不健康，但仍然返回所有端点（让重试机制处理）
                if self._last_health_summary != current_summary:
                    print(f"[WARN] 所有端点都不健康，但仍将尝试使用（可能服务正在启动）")
                self._last_health_summary = current_summary
                return self.endpoints

        # 使用缓存的健康状态
        healthy_endpoints = [
            endpoint for endpoint in self.endpoints
            if self._health_status.get(endpoint, (True, 0.0, 0))[0]
        ]
        # 如果缓存的健康端点为空，进行全量检查
        if not healthy_endpoints:
            return self._get_healthy_endpoints(force_check=True)

        return healthy_endpoints

    def _check_endpoint_health(self, endpoint: str) -> bool:
        """
        检查单个端点的健康状态

        Args:
        endpoint: 服务端点URL

        Returns:
        True if healthy, False otherwise
        """
        try:
            health_url = f"{endpoint}/health"
            response = requests.get(
                health_url,
                timeout=self.health_check_timeout,
                headers={"Content-Type": "application/json"}
            )
            is_healthy = response.status_code == 200
        except Exception as e:
            is_healthy = False

        # 更新健康状态
        current_time = time.time()
        was_healthy, last_check, failures = self._health_status.get(endpoint, (True, 0.0, 0))

        if is_healthy:
            consecutive_failures = 0
            # 健康恢复信息已在健康检查结果中显示，这里不再重复打印
        else:
            consecutive_failures = failures + 1
            max_failure_count = 1000
            if consecutive_failures > max_failure_count:
                consecutive_failures = max_failure_count
            # 当达到上限时，强制进行一次健康检查（给端点恢复的机会）
            if consecutive_failures == max_failure_count:
                print(f"[INFO] 端点失败计数达到上限 ({max_failure_count})，将定期检查恢复: {endpoint}")

            # 允许偶尔失败（连续失败阈值可配置，默认5次）
            # 服务在处理请求时可能暂时无法响应健康检查，网络抖动也可能导致误判
            # 在分布式环境中，节点不会真正断线，只是通信问题导致暂时无法响应
            # 提高阈值可以减少误判，避免频繁标记为不健康
            health_check_failure_threshold = _get_env_int("API_HEALTH_CHECK_FAILURE_THRESHOLD", 5)
            if consecutive_failures < health_check_failure_threshold:
                # 偶尔失败，仍然认为健康（避免误判）
                is_healthy = True
                consecutive_failures = failures + 1  # 保留失败计数，但不标记为不健康
            # 如果从健康变为不健康，打印警告（但减少日志频率）
            if was_healthy and consecutive_failures >= health_check_failure_threshold:
                # 只在阈值达到时打印一次，避免重复日志
                if consecutive_failures == health_check_failure_threshold:
                    print(f"[WARN] 端点变为不健康: {endpoint} (连续失败: {consecutive_failures}次，阈值: {health_check_failure_threshold})")

        self._health_status[endpoint] = (is_healthy, current_time, consecutive_failures)

        return is_healthy

    def _get_next_endpoint(self) -> str:
        """
        轮询获取下一个健康端点（优化负载均衡逻辑）

        Returns:
        健康的服务端点URL
        """
        healthy_endpoints = self._get_healthy_endpoints()

        if not healthy_endpoints:
            # 没有健康端点，返回所有端点（让重试机制处理）
            healthy_endpoints = self.endpoints

        # 直接从健康端点列表中轮询，确保负载均衡
        # 使用健康端点列表的索引，而不是原始端点列表的索引
        if healthy_endpoints:
            # 线程安全地更新 round-robin 指针，避免并发线程拿到相同 index 导致热点
            with self._rr_lock:
                healthy_index = self._current_index % len(healthy_endpoints)
                endpoint = healthy_endpoints[healthy_index]
                self._current_index = (self._current_index + 1) % len(healthy_endpoints)
        else:
            # 如果没有健康端点，回退到原始端点列表
            with self._rr_lock:
                endpoint = self.endpoints[self._current_index % len(self.endpoints)]
                self._current_index = (self._current_index + 1) % len(self.endpoints)

        return endpoint

    def _post_with_retry(self, path: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        带重试和健康检查的POST请求

        Args:
        path: API路径
        json_data: 请求JSON数据

        Returns:
        响应JSON数据

        Raises:
        RuntimeError: 所有重试都失败
        """
        last_error = None
        tried_endpoints: Set[str] = set()
        max_endpoint_attempts = len(self.endpoints) * 2  # 最多尝试所有端点2轮

        for attempt in range(max_endpoint_attempts):
            endpoint = self._get_next_endpoint()

            # 如果已经尝试过所有端点，进行全量健康检查
            if len(tried_endpoints) >= len(self.endpoints):
                tried_endpoints.clear()  # 重置，开始新的一轮
                # 避免频繁输出日志，只在间隔时间后打印
                current_time = time.time()
                if current_time - self._last_full_check_log_time >= self._full_check_log_interval:
                    print(f"[INFO] 已尝试所有端点，进行全量健康检查...", flush=True)
                self._last_full_check_log_time = current_time

                # 强制进行全量健康检查，给端点恢复的机会
                healthy_endpoints = self._get_healthy_endpoints(force_check=True)

                # 如果仍然没有健康端点，等待一段时间再重试（避免立即重试导致资源浪费）
                if not healthy_endpoints or len(healthy_endpoints) == 0:
                    if attempt < max_endpoint_attempts - 1:
                        wait_time = min(5, 2 ** (attempt // len(self.endpoints)))  # 指数退避，最多等待5秒
                        print(f"[WARN] 所有端点都不健康，等待 {wait_time} 秒后重试...", flush=True)
                        time.sleep(wait_time)

            # 如果端点不健康（启用健康检查时），直接跳过，不尝试请求
            if self.enable_health_check:
                is_healthy, _, consecutive_failures = self._health_status.get(
                    endpoint, (True, 0.0, 0)
                )

                # 获取健康检查失败阈值（与 _check_endpoint_health 保持一致）
                health_check_failure_threshold = _get_env_int("API_HEALTH_CHECK_FAILURE_THRESHOLD", 5)

                # 如果端点不健康，直接跳过（不等待超时）
                # 只有当连续失败次数 >= 阈值时，才认为端点不健康
                if not is_healthy and consecutive_failures >= health_check_failure_threshold:
                    # 如果连续失败次数较少（<阈值+3），快速检查一次看是否恢复
                    if consecutive_failures < health_check_failure_threshold + 3:
                        if self._check_endpoint_health(endpoint):
                            # 端点已恢复，继续使用
                            pass
                        else:
                            # 端点仍然不健康，跳过
                            tried_endpoints.add(endpoint)
                            last_error = f"Endpoint {endpoint} is unhealthy (consecutive failures: {consecutive_failures})"
                            print(f"[WARN] 跳过不健康的端点: {endpoint} (连续失败: {consecutive_failures})")
                            continue
                    else:
                        # 连续失败次数过多，需要定期检查恢复
                        # 根据失败次数动态调整检查频率
                        # - 失败次数 < 100: 每5次检查一次
                        # - 失败次数 < 500: 每20次检查一次
                        # - 失败次数 >= 500: 每50次检查一次（避免过度检查）
                        # 同时，如果距离上次检查超过60秒，强制检查一次
                        current_time = time.time()
                        last_check_time = self._health_status.get(endpoint, (False, 0.0, 0))[1]
                        time_since_last_check = current_time - last_check_time

                        # 动态检查频率
                        if consecutive_failures < 100:
                            check_interval = 5
                        elif consecutive_failures < 500:
                            check_interval = 20
                        else:
                            check_interval = 50

                        # 决定是否检查：按失败次数间隔 或 距离上次检查超过60秒
                        should_check = (
                            (consecutive_failures % check_interval == 0) or
                            (time_since_last_check > 60.0)
                        )

                        should_log = False

                        # 当失败次数 > 50 时，每50次失败才打印一次日志
                        if consecutive_failures > 50:
                            log_interval = self._unhealthy_log_interval
                            log_count = self._unhealthy_endpoint_log_count.get(endpoint, 0)
                            # 计算应该打印日志的次数（每50次失败打印一次）
                            expected_log_count = consecutive_failures // log_interval
                            if expected_log_count > log_count:
                                should_log = True
                                self._unhealthy_endpoint_log_count[endpoint] = expected_log_count
                        else:
                            # 失败次数 <= 50 时，正常打印日志
                            should_log = True

                        if should_check:
                            if self._check_endpoint_health(endpoint):
                                # 端点已恢复，继续使用
                                # 清除日志计数，恢复正常日志
                                self._unhealthy_endpoint_log_count.pop(endpoint, None)
                                print(f"[INFO] 端点已恢复: {endpoint} (之前连续失败: {consecutive_failures}次)")
                                pass
                            else:
                                tried_endpoints.add(endpoint)
                                last_error = f"Endpoint {endpoint} is unhealthy (consecutive failures: {consecutive_failures})"
                                if should_log:
                                    print(f"[WARN] 跳过不健康的端点: {endpoint} (连续失败: {consecutive_failures})")
                                continue
                        else:
                            tried_endpoints.add(endpoint)
                            last_error = f"Endpoint {endpoint} is unhealthy (consecutive failures: {consecutive_failures})"
                            # 只在应该打印日志时才打印
                            if should_log:
                                print(f"[WARN] 跳过不健康的端点: {endpoint} (连续失败: {consecutive_failures})")
                            continue

            url = f"{endpoint}{path}"
            tried_endpoints.add(endpoint)

            try:
                response = requests.post(
                    url,
                    json=json_data,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    # 请求成功，标记端点健康（被动健康检查）
                    if self.enable_health_check:
                        current_time = time.time()
                        self._health_status[endpoint] = (True, current_time, 0)
                        self._last_request_time = current_time  # 更新最后请求时间
                    return response.json()
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    print(f"[WARN] Request failed to {endpoint} (attempt {attempt + 1}/{max_endpoint_attempts}): {last_error}")

                    # 如果是5xx错误，标记端点不健康
                    if self.enable_health_check and 500 <= response.status_code < 600:
                        current_time = time.time()
                        self._last_request_time = current_time  # 更新最后请求时间
                        _, _, failures = self._health_status.get(endpoint, (True, 0.0, 0))
                        self._health_status[endpoint] = (False, current_time, failures + 1)

            except requests.exceptions.Timeout as e:
                last_error = f"Timeout after {self.timeout}s: {str(e)}"
                print(f"[WARN] Request timeout to {endpoint} (attempt {attempt + 1}/{max_endpoint_attempts}): {last_error}")

                # 超时后快速检查一次（被动健康检查）
                if self.enable_health_check:
                    current_time = time.time()
                    self._last_request_time = current_time  # 更新最后请求时间
                    # 快速检查一次（使用较短的超时时间，避免等待太久）
                    quick_check_timeout = min(5, self.health_check_timeout)
                    try:
                        health_url = f"{endpoint}/health"
                        quick_response = requests.get(health_url, timeout=quick_check_timeout)
                        is_healthy = quick_response.status_code == 200
                    except:
                        is_healthy = False

                    _, _, failures = self._health_status.get(endpoint, (True, 0.0, 0))
                    if is_healthy:
                        # 快速检查成功，可能是业务请求超时但服务正常，不标记为不健康
                        # 但增加失败计数，连续失败3次才标记为不健康
                        if failures + 1 >= 3:
                            self._health_status[endpoint] = (False, current_time, failures + 1)
                        else:
                            self._health_status[endpoint] = (True, current_time, failures + 1)
                    else:
                        # 快速检查失败，标记为不健康
                        self._health_status[endpoint] = (False, current_time, failures + 1)

            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {str(e)}"
                print(f"[WARN] Connection error to {endpoint} (attempt {attempt + 1}/{max_endpoint_attempts}): {last_error}")

                # 连接错误直接标记为不健康（不需要快速检查，连接错误说明服务不可用）
                if self.enable_health_check:
                    current_time = time.time()
                    self._last_request_time = current_time  # 更新最后请求时间
                    self._health_status[endpoint] = (False, current_time, 10)  # 连接错误，设置较高失败计数

            except Exception as e:
                last_error = str(e)
                print(f"[WARN] Request failed to {endpoint} (attempt {attempt + 1}/{max_endpoint_attempts}): {last_error}")

            # 等待后重试（指数退避，但不超过所有端点轮询）
            if attempt < max_endpoint_attempts - 1:
                sleep_time = min(2 ** (attempt % 3), 8)  # 最多等待8秒
                time.sleep(sleep_time)

        # 所有重试都失败
        error_msg = f"Request failed after {max_endpoint_attempts} attempts to different endpoints. Last error: {last_error}"

        # 打印健康状态摘要（优化：减少详细输出，只在关键信息时打印）
        if self.enable_health_check:
            healthy_count = sum(1 for h, _, _ in self._health_status.values() if h)
            total_count = len(self.endpoints)
            print(f"[ERROR] {error_msg}")
            print(f"[ERROR] 端点健康状态: {healthy_count}/{total_count} 健康")

            # 只打印不健康的端点（减少日志量）
            unhealthy_endpoints_info = []
            for endpoint, (is_healthy, last_check, failures) in self._health_status.items():
                if not is_healthy:
                    unhealthy_endpoints_info.append((endpoint, failures, last_check))

            # 如果有很多不健康端点，只打印前5个，避免日志过长
            if len(unhealthy_endpoints_info) > 5:
                for endpoint, failures, last_check in unhealthy_endpoints_info[:5]:
                    time_ago = time.time() - last_check
                    print(f"[ERROR]   ❌ {endpoint} (failures: {failures}, last_check: {time_ago:.1f}s ago)")
                print(f"[ERROR]   ... 还有 {len(unhealthy_endpoints_info) - 5} 个不健康端点")
            else:
                for endpoint, failures, last_check in unhealthy_endpoints_info:
                    time_ago = time.time() - last_check
                    print(f"[ERROR]   ❌ {endpoint} (failures: {failures}, last_check: {time_ago:.1f}s ago)")

        raise RuntimeError(error_msg)


class ImageEditClient(ServiceClient):
    """图像编辑服务客户端（支持BAGEL和Qwen-Image-Edit）"""

    def __init__(self, *args, **kwargs):
        """初始化图像编辑客户端"""
        super().__init__(*args, **kwargs)
        # 缓存每个端点的服务类型：endpoint -> model_type ("bagel" 或 "qwen_image_edit")
        self._model_types: Dict[str, str] = {}

    def _detect_model_type(self, endpoint: str) -> str:
        """
        检测端点的服务类型（BAGEL 或 Qwen-Image-Edit）
        
        Args:
            endpoint: 服务端点URL
            
        Returns:
            "bagel" 或 "qwen_image_edit"
        """
        # 如果已缓存，直接返回
        if endpoint in self._model_types:
            return self._model_types[endpoint]
        
        # 通过健康检查端点检测服务类型
        try:
            health_url = f"{endpoint}/health"
            response = requests.get(
                health_url,
                timeout=self.health_check_timeout,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                data = response.json()
                model_type = data.get("model_type", "").lower()
                if "qwen" in model_type:
                    detected_type = "qwen_image_edit"
                else:
                    detected_type = "bagel"  # 默认为bagel
            else:
                detected_type = "bagel"  # 默认为bagel
        except Exception:
            # 如果检测失败，默认为bagel（向后兼容）
            detected_type = "bagel"
        
        # 缓存结果
        self._model_types[endpoint] = detected_type
        return detected_type

    def edit_image(
        self,
        image: Image.Image,
        edit_prompt: str,
        resolution: int = 1024,
        num_timesteps: int = 50,
        cfg_text_scale: float = None,
        cfg_img_scale: float = None,
        cfg_scale: float = None,  # 向后兼容，等同于cfg_text_scale
        cfg_interval: list = None,
        timestep_shift: float = 3.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "text_channel",
        # Taylor加速参数（可选，仅BAGEL）
        enable_taylorseer: bool = None,
        fresh_threshold: int = None,
        max_order: int = None,
        first_enhance: int = None,
        # 分辨率缩放参数（可选，两个服务都支持）
        resolution_scale: float = None,
        # Qwen专用参数
        guidance_scale: float = 1.0,  # Qwen的guidance_scale（对应BAGEL的cfg_text_scale）
        true_cfg_scale: float = 4.0,  # Qwen的true_cfg_scale
        negative_prompt: str = None,  # Qwen的negative_prompt
        steps_computation_mask: list = None,  # Qwen的steps_computation_mask
        steps_computation_policy: str = "static",  # Qwen的steps_computation_policy
        use_scm: bool = False,  # Qwen的SCM加速
        scm_policy: str = None,  # Qwen的SCM策略
        # 显式指定服务类型（可选，如果为None则自动检测）
        model_type: Optional[str] = None,
    ) -> Image.Image:
        """
        编辑图像（支持BAGEL和Qwen服务）

        Args:
            image: 原始图像
            edit_prompt: 编辑指令
            resolution: 输出分辨率（BAGEL使用，Qwen忽略）
            num_timesteps: 去噪步数（BAGEL使用，Qwen对应num_inference_steps）
            cfg_text_scale: CFG text scale（BAGEL使用）
            cfg_img_scale: CFG img scale（BAGEL使用）
            cfg_scale: CFG scale（向后兼容，等同于cfg_text_scale，BAGEL使用）
            cfg_interval: CFG interval（BAGEL使用）
            timestep_shift: Timestep shift（BAGEL使用）
            cfg_renorm_min: CFG renormalization minimum（BAGEL使用）
            cfg_renorm_type: CFG renormalization type（BAGEL使用）
            enable_taylorseer: 是否启用Taylor加速（BAGEL使用，None表示使用服务端默认值）
            fresh_threshold: Taylor加速参数（BAGEL使用，None表示使用服务端默认值）
            max_order: Taylor加速参数（BAGEL使用，None表示使用服务端默认值）
            first_enhance: Taylor加速参数（BAGEL使用，None表示使用服务端默认值）
            resolution_scale: 分辨率缩放因子 (0, 1.0]，两个服务都支持（None表示使用服务端默认值）
            guidance_scale: Qwen的guidance_scale（如果提供，优先使用；否则使用cfg_text_scale或cfg_scale）
            true_cfg_scale: Qwen的true_cfg_scale（None表示使用服务端默认值）
            negative_prompt: Qwen的negative_prompt（None表示使用服务端默认值）
            use_scm: Qwen的SCM加速（None表示使用服务端默认值）
            scm_policy: Qwen的SCM策略（None表示使用服务端默认值）
            model_type: 显式指定服务类型（"bagel" 或 "qwen_image_edit"），如果为None则自动检测

        Returns:
            编辑后的图像
        """
        # 编码图像为base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # 检测服务类型（如果未指定）
        if model_type is None:
            # 获取第一个健康端点并检测其类型（不改变轮询索引）
            healthy_endpoints = self._get_healthy_endpoints()
            if healthy_endpoints:
                # 使用第一个健康端点检测类型（不改变轮询状态）
                first_endpoint = healthy_endpoints[0]
                detected_type = self._detect_model_type(first_endpoint)
            else:
                # 如果没有健康端点，使用第一个端点检测
                if self.endpoints:
                    detected_type = self._detect_model_type(self.endpoints[0])
                else:
                    detected_type = "bagel"  # 默认
        else:
            detected_type = model_type.lower()
            if detected_type == "qwen":
                detected_type = "qwen_image_edit"

        # 根据服务类型构建不同的请求参数
        if detected_type == "qwen_image_edit":
            # Qwen服务参数
            json_data = {
                "image": image_b64,
                "edit_prompt": edit_prompt,
            }
            
            # num_inference_steps
            json_data["num_inference_steps"] = num_timesteps
            json_data["steps_computation_mask"] = steps_computation_mask if steps_computation_mask is not None else [1 if (i-1) % 3 == 0 or i<5 else 0 for i in range(num_timesteps)]
            json_data["true_cfg_scale"] = true_cfg_scale
            json_data["guidance_scale"] = guidance_scale
            json_data["steps_computation_policy"] = steps_computation_policy
            # negative_prompt
            if negative_prompt is not None:
                json_data["negative_prompt"] = negative_prompt
            
            # SCM参数
            if use_scm is not None:
                json_data["use_scm"] = use_scm
            if scm_policy is not None:
                json_data["scm_policy"] = scm_policy
            
            # resolution_scale（两个服务都支持）
            if resolution_scale is not None:
                json_data["resolution_scale"] = resolution_scale
        else:
            # BAGEL服务参数（保持向后兼容）
            json_data = {
                "image": image_b64,
                "edit_prompt": edit_prompt,
                "resolution": resolution,
                "num_timesteps": num_timesteps,
                "timestep_shift": timestep_shift,
                "cfg_renorm_min": cfg_renorm_min,
                "cfg_renorm_type": cfg_renorm_type,
            }

            # 处理cfg_scale向后兼容：优先使用cfg_text_scale，如果没有则使用cfg_scale
            if cfg_text_scale is not None:
                json_data["cfg_text_scale"] = cfg_text_scale
            elif cfg_scale is not None:
                json_data["cfg_scale"] = cfg_scale  # 向后兼容，服务端支持

            # 只添加非None的参数
            if cfg_img_scale is not None:
                json_data["cfg_img_scale"] = cfg_img_scale
            if cfg_interval is not None:
                json_data["cfg_interval"] = cfg_interval
            if enable_taylorseer is not None:
                json_data["enable_taylorseer"] = enable_taylorseer
            if fresh_threshold is not None:
                json_data["fresh_threshold"] = fresh_threshold
            if max_order is not None:
                json_data["max_order"] = max_order
            if first_enhance is not None:
                json_data["first_enhance"] = first_enhance
            if resolution_scale is not None:
                json_data["resolution_scale"] = resolution_scale

        # 发送请求
        result = self._post_with_retry("/edit", json_data)

        if not result.get("success"):
            raise RuntimeError(f"Image edit failed: {result.get('error')}")

        # 解码返回的图像
        edited_image_b64 = result["image"]
        edited_image_bytes = base64.b64decode(edited_image_b64)
        edited_image = Image.open(io.BytesIO(edited_image_bytes)).convert("RGB")
    
        return edited_image


class RewardClient(ServiceClient):
    """奖励计算服务客户端"""

    def compute_reward(
        self,
        image: Image.Image,
        prompt: str,
        reward_type: str = "clip",
        generated_qa: Optional[Dict[str, Any]] = None,
        # SAM3 专用参数
        category: Optional[str] = None,
        ground_truth: Optional[str] = None,
        bpe_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        metadata_jsonl_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        统一的奖励计算接口

        Args:
            image: 图像
            prompt: 文本提示
            reward_type: 奖励类型 ("clip", "sam3")
            generated_qa: 可选的分解问题，格式为 {"yn_question_list": [...]}
            category: SAM3 专用 - 图像类别
            ground_truth: SAM3 专用 - 地面真值
            bpe_path: SAM3 专用 - BPE路径
            checkpoint_path: SAM3 专用 - 检查点路径
            metadata_jsonl_path: SAM3 专用 - 元数据JSONL路径

        Returns:
            奖励结果字典
        """
        # 编码图像
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # 构建基础请求
        json_data = {
            "image": image_b64,
            "prompt": prompt,
            "reward_type": reward_type,
        }
        if generated_qa is not None:
            json_data["generated_qa"] = generated_qa

        # SAM3 专用参数
        if reward_type.lower() == "sam3":
            if not category or not ground_truth:
                raise ValueError("SAM3 奖励需要 category 和 ground_truth 参数")

            json_data.update({
                "category": category,
                "ground_truth": ground_truth,
            })
            if bpe_path is not None:
                json_data["bpe_path"] = bpe_path
            if checkpoint_path is not None:
                json_data["checkpoint_path"] = checkpoint_path
            if metadata_jsonl_path is not None:
                json_data["metadata_jsonl_path"] = metadata_jsonl_path

            # SAM3 使用专用接口
            result = self._post_with_retry("/compute_sam3_reward", json_data)
        else:
            # 普通奖励使用通用接口
            result = self._post_with_retry("/compute_reward", json_data)

        if not result.get("success"):
            error_msg = result.get('error') or result.get('error_message') or "Unknown error"
            raise RuntimeError(f"Reward computation failed: {error_msg}")

        return result


# 全局变量：存储配置文件的修改时间和客户端实例
_config_file_mtime: Dict[str, float] = {}
_global_edit_client: Optional[ImageEditClient] = None
_global_reward_client: Optional[RewardClient] = None
_global_sam3_client: Optional[RewardClient] = None

def _get_config_file_path() -> Optional[str]:
    """获取 service_endpoints.env 文件路径"""
    # 尝试多个可能的路径
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "config", "service_endpoints.env"),
        os.path.join(os.environ.get("CONFIG_DIR", ""), "service_endpoints.env") if os.environ.get("CONFIG_DIR") else None
    ]

    for path in possible_paths:
        if path and os.path.exists(path):
            return os.path.abspath(path)

    return None


def _reload_config_file(config_file: str) -> bool:
    """
    重新加载配置文件到环境变量

    支持标准的 bash env 文件格式：
    - export KEY="VALUE"
    - export KEY='VALUE'
    - export KEY=VALUE
    - 支持注释行（以 # 开头）

    Args:
        config_file: 配置文件路径

    Returns:
        True if reloaded, False if file not changed
    """
    try:
        current_mtime = os.path.getmtime(config_file)
        last_mtime = _config_file_mtime.get(config_file, 0.0)

        # 如果文件未修改，不需要重新加载
        if current_mtime <= last_mtime:
            return False

        # 读取配置文件并更新环境变量
        with open(config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # 跳过注释和空行
                if not line or line.startswith('#'):
                    continue

                # 只处理 export 语句
                if not line.startswith('export '):
                    continue

                # 解析 export KEY="VALUE" 或 export KEY=VALUE
                export_part = line[7:].strip()  # 去掉 'export '
                if '=' not in export_part:
                    continue

                # 找到第一个 = 的位置（作为分隔符）
                eq_pos = export_part.find('=')
                key = export_part[:eq_pos].strip()
                value_part = export_part[eq_pos+1:].strip()

                # 处理引号包裹的值
                if value_part.startswith('"') and value_part.endswith('"'):
                    # 双引号：去掉引号，保留内部内容
                    value = value_part[1:-1]
                elif value_part.startswith("'") and value_part.endswith("'"):
                    # 单引号：去掉引号，保留内部内容
                    value = value_part[1:-1]
                else:
                    # 无引号：直接使用，但需要去掉可能的注释
                    value = value_part
                    # 如果值中包含 #，且不在引号内，则去掉注释部分
                    if '#' in value:
                        # 简单处理：如果 # 不在引号内，则去掉注释
                        # 注意：这里假设值中不包含未转义的引号
                        comment_pos = value.find('#')
                        if comment_pos >= 0:
                            value = value[:comment_pos].strip()

                # 设置环境变量
                if key:
                    os.environ[key] = value

        _config_file_mtime[config_file] = current_mtime
        print(f"[INFO] 配置文件已重新加载: {config_file}")
        return True
    except Exception as e:
        print(f"[WARNING] 重新加载配置文件失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def _check_and_reload_config(
    edit_client: Optional[ImageEditClient],
    reward_client: Optional[RewardClient],
    sam3_client: Optional[RewardClient]  # 添加 SAM3 客户端参数
) -> bool:
    """
    检查配置文件是否修改，如果修改则重新加载并更新客户端
    """
    config_file = _get_config_file_path()
    if not config_file:
        return False

    # 检查文件是否修改
    if not _reload_config_file(config_file):
        return False

    # 重新读取环境变量
    edit_endpoints_str = os.environ.get("EDIT_SERVER_ENDPOINTS", "")
    reward_endpoints_str = os.environ.get("REWARD_SERVER_ENDPOINTS", "")
    sam3_endpoints_str = os.environ.get("SAM3_REWARD_SERVER_ENDPOINTS", "")

    updated = False

    # 更新图像编辑客户端
    if edit_client and edit_endpoints_str:
        new_endpoints = [e.strip() for e in edit_endpoints_str.split(",") if e.strip()]
        if new_endpoints:
            if edit_client.update_endpoints(new_endpoints):
                updated = True

    # 更新奖励计算客户端
    if reward_client and reward_endpoints_str:
        new_endpoints = [e.strip() for e in reward_endpoints_str.split(",") if e.strip()]
        if new_endpoints:
            if reward_client.update_endpoints(new_endpoints):
                updated = True
    # 更新 SAM3 客户端
    if sam3_client and sam3_endpoints_str:
        new_endpoints = [e.strip() for e in sam3_endpoints_str.split(",") if e.strip()]
        if new_endpoints:
            if sam3_client.update_endpoints(new_endpoints):
                updated = True
    return updated


def create_clients_from_env(check_config_file: bool = True) -> tuple[
    Optional[ImageEditClient],
    Optional[RewardClient],
    Optional[RewardClient]  # 第三个返回值是 SAM3 客户端
]:
    """
    从环境变量创建客户端实例（支持配置文件热重载）
    返回: (edit_client, reward_client, sam3_client)
    """
    global _global_edit_client, _global_reward_client, _global_sam3_client

    # 如果已存在客户端且启用热重载，检查配置文件
    if check_config_file and (_global_edit_client or _global_reward_client or _global_sam3_client):
        _check_and_reload_config(_global_edit_client, _global_reward_client, _global_sam3_client)
        return _global_edit_client, _global_reward_client, _global_sam3_client

    edit_client: Optional[ImageEditClient] = None
    reward_client: Optional[RewardClient] = None
    sam3_client: Optional[RewardClient] = None

    # 如果启用热重载，先尝试从配置文件加载
    if check_config_file:
        config_file = _get_config_file_path()
        if config_file:
            _reload_config_file(config_file)

    # 从环境变量读取配置（默认180秒，3分钟，与当前api_client.py保持一致）
    timeout = _get_env_int("API_REQUEST_TIMEOUT", 180)
    max_retries = _get_env_int("API_MAX_RETRIES", 3)
    health_check_timeout = _get_env_int("API_HEALTH_CHECK_TIMEOUT", 5)
    health_check_interval = _get_env_int("API_HEALTH_CHECK_INTERVAL", 600)  # backup文件默认600秒
    enable_health_check = _get_env_bool("API_ENABLE_HEALTH_CHECK", True)

    # 图像编辑客户端
    edit_endpoints = os.environ.get("EDIT_SERVER_ENDPOINTS", "")
    if edit_endpoints:
        endpoints = [e.strip() for e in edit_endpoints.split(",") if e.strip()]
        if endpoints:
            edit_client = ImageEditClient(
                endpoints,
                timeout=timeout,
                max_retries=max_retries,
                health_check_timeout=health_check_timeout,
                health_check_interval=health_check_interval,
                enable_health_check=enable_health_check
            )
            # 只在真正创建新客户端时打印（避免重复创建时的噪音）
            if not _global_edit_client:
                health_status = "启用" if enable_health_check else "禁用"
                print(f"✅ Image Edit Client created, endpoints: {len(endpoints)}, health check: {health_status}")

    # 奖励计算客户端（CLIP/self_reward）
    reward_endpoints = os.environ.get("REWARD_SERVER_ENDPOINTS", "")
    if reward_endpoints:
        endpoints = [e.strip() for e in reward_endpoints.split(",") if e.strip()]
        if endpoints:
            reward_client = RewardClient(
                endpoints,
                timeout=timeout,
                max_retries=max_retries,
                health_check_timeout=health_check_timeout,
                health_check_interval=health_check_interval,
                enable_health_check=enable_health_check
            )
            # 只在真正创建新客户端时打印（避免重复创建时的噪音）
            if not _global_reward_client:
                health_status = "启用" if enable_health_check else "禁用"
                print(f"✅ Reward Client created, endpoints: {len(endpoints)}, health check: {health_status}")

    sam3_endpoints = os.environ.get("SAM3_REWARD_SERVER_ENDPOINTS", "")
    if sam3_endpoints:
        endpoints = [e.strip() for e in sam3_endpoints.split(",") if e.strip()]
        if endpoints:
            sam3_client = RewardClient(
                endpoints,
                timeout=timeout,
                max_retries=max_retries,
                health_check_timeout=health_check_timeout,
                health_check_interval=health_check_interval,
                enable_health_check=enable_health_check
            )
            # 只在真正创建新客户端时打印（避免重复创建时的噪音）
            if not _global_sam3_client:
                health_status = "启用" if enable_health_check else "禁用"
                print(f"✅ SAM3 Reward Client created, endpoints: {len(endpoints)}, health check: {health_status}")

    # 保存全局引用（用于热重载）
    if check_config_file:
        _global_edit_client = edit_client
        _global_reward_client = reward_client
        _global_sam3_client = sam3_client

    return edit_client, reward_client, sam3_client


# 示例用法
if __name__ == "__main__":
    import os
    from PIL import Image

    # 方式1: 从环境变量创建
    edit_client, reward_client, sam3_client = create_clients_from_env()

    # 方式2: 手动创建
    if not edit_client:
        edit_client = ImageEditClient([
            "http://node2:5001",
            "http://node2:5002",
            "http://node2:5003",
            "http://node2:5004",
        ])

    if not reward_client:
        reward_client = RewardClient([
            "http://node3:5002",
            "http://node3:5003",
        ])

    # 测试图像编辑
    if edit_client:
        print("\n测试图像编辑服务...")
        test_image = Image.new("RGB", (512, 512), color=(255, 0, 0))
        try:
            edited = edit_client.edit_image(
                test_image,
                "Make it blue",
                resolution=512,
                num_timesteps=20,
            )
            print(f"✅ 编辑成功，图像尺寸: {edited.size}")
        except Exception as e:
            print(f"❌ 编辑失败: {e}")

    # 测试奖励计算
    if reward_client:
        print("\n测试奖励计算服务...")
        test_image = Image.new("RGB", (512, 512), color=(0, 255, 0))
        try:
            result = reward_client.compute_reward(
                test_image,
                "A green square",
                reward_type="clip"
            )
            print(f"✅ 奖励计算成功，分数: {result['score']:.4f}")
        except Exception as e:
            print(f"❌ 奖励计算失败: {e}")
