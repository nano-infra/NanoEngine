#!/usr/bin/env python3
"""
NanoRoute 字符串评测脚本
支持流式和非流式请求的性能测试
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import aiohttp


@dataclass
class BenchmarkResult:
    """单次请求的结果"""

    request_id: int
    success: bool
    latency_ms: float
    first_token_ms: Optional[float] = None  # 流式请求的首 token 延迟
    total_tokens: int = 0
    error_msg: Optional[str] = None


class NanoRouteBenchmark:
    """NanoRoute 性能评测类"""

    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/v1/chat/completions"

    async def single_request(
        self,
        messages: List[Dict[str, str]],
        model: str = "opt-1.3b",
        max_tokens: int = 16,
        stream: bool = False,
        request_id: int = 0,
    ) -> BenchmarkResult:
        """发送单个请求并测量性能"""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        start_time = time.time()
        first_token_time = None
        total_tokens = 0
        success = False
        error_msg = None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status != 200:
                        error_msg = f"HTTP {response.status}: {await response.text()}"
                        latency_ms = (time.time() - start_time) * 1000
                        return BenchmarkResult(
                            request_id=request_id,
                            success=False,
                            latency_ms=latency_ms,
                            error_msg=error_msg,
                        )

                    if stream:
                        # 流式响应处理 (SSE 格式)
                        buffer = ""
                        async for chunk in response.content.iter_chunked(1024):
                            buffer += chunk.decode("utf-8", errors="ignore")

                            # 按行处理
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()

                                if not line or line.startswith(":"):
                                    continue

                                if line.startswith("data: "):
                                    data_str = line[6:]  # 去掉 'data: ' 前缀

                                    if data_str == "[DONE]":
                                        break

                                    try:
                                        data = json.loads(data_str)
                                        if first_token_time is None:
                                            first_token_time = time.time()

                                        # 统计 token 数量
                                        choices = data.get("choices", [])
                                        if choices and "delta" in choices[0]:
                                            delta = choices[0]["delta"]
                                            if "content" in delta and delta["content"]:
                                                total_tokens += 1  # 简化统计
                                    except json.JSONDecodeError:
                                        continue

                        latency_ms = (time.time() - start_time) * 1000
                        first_token_ms = (
                            (first_token_time - start_time) * 1000
                            if first_token_time
                            else None
                        )
                        success = True
                    else:
                        # 非流式响应处理
                        result = await response.json()
                        latency_ms = (time.time() - start_time) * 1000
                        first_token_ms = None  # 非流式请求没有首 token 延迟

                        # 统计 token 数量（简化处理）
                        if "choices" in result and len(result["choices"]) > 0:
                            content = (
                                result["choices"][0]
                                .get("message", {})
                                .get("content", "")
                            )
                            total_tokens = len(content.split())  # 简化统计

                        success = True

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            first_token_ms = None
            error_msg = str(e)
            success = False

        return BenchmarkResult(
            request_id=request_id,
            success=success,
            latency_ms=latency_ms,
            first_token_ms=first_token_ms,
            total_tokens=total_tokens,
            error_msg=error_msg,
        )

    async def concurrent_benchmark(
        self,
        messages: List[Dict[str, str]],
        num_requests: int = 10,
        concurrency: int = 1,
        model: str = "opt-1.3b",
        max_tokens: int = 16,
        stream: bool = False,
    ) -> List[BenchmarkResult]:
        """并发性能测试"""
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(request_id: int):
            async with semaphore:
                return await self.single_request(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    stream=stream,
                    request_id=request_id,
                )

        tasks = [bounded_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        return list(results)

    def print_statistics(
        self, results: List[BenchmarkResult], total_time_seconds: float = None
    ):
        """打印统计信息"""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print("\n" + "=" * 60)
        print("性能评测结果")
        print("=" * 60)
        print(f"总请求数: {len(results)}")
        print(f"成功请求: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"失败请求: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")

        if failed:
            print("\n失败请求详情:")
            for r in failed:
                print(f"  请求 #{r.request_id}: {r.error_msg}")

        if successful:
            latencies = [r.latency_ms for r in successful]
            print(f"\n延迟统计 (ms):")
            print(f"  平均: {statistics.mean(latencies):.2f}")
            print(f"  中位数: {statistics.median(latencies):.2f}")
            print(f"  最小值: {min(latencies):.2f}")
            print(f"  最大值: {max(latencies):.2f}")
            if len(latencies) > 1:
                print(f"  标准差: {statistics.stdev(latencies):.2f}")

            # P50, P90, P95, P99
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            print(f"\n延迟分位数 (ms):")
            print(f"  P50: {sorted_latencies[int(n*0.50)]:.2f}")
            print(f"  P90: {sorted_latencies[int(n*0.90)]:.2f}")
            print(f"  P95: {sorted_latencies[int(n*0.95)]:.2f}")
            print(f"  P99: {sorted_latencies[int(n*0.99)]:.2f}")

            # 吞吐量计算
            if total_time_seconds and total_time_seconds > 0:
                # 使用实际总耗时计算吞吐量
                qps = len(successful) / total_time_seconds
                print(f"\n吞吐量 (QPS): {qps:.2f} 请求/秒")
            else:
                # 使用平均延迟估算
                avg_latency_sec = statistics.mean(latencies) / 1000
                if avg_latency_sec > 0:
                    qps = 1.0 / avg_latency_sec
                    print(f"\n吞吐量 (QPS, 估算): {qps:.2f} 请求/秒")

            # TPS (Tokens Per Second) 和 TPOT (Time Per Output Token) 计算
            total_tokens = sum(r.total_tokens for r in successful)
            if total_tokens > 0:
                if total_time_seconds and total_time_seconds > 0:
                    tps = total_tokens / total_time_seconds
                    print(f"吞吐量 (TPS): {tps:.2f} tokens/秒")
                else:
                    # 使用平均延迟估算
                    avg_latency_sec = statistics.mean(latencies) / 1000
                    if avg_latency_sec > 0:
                        avg_tokens_per_request = total_tokens / len(successful)
                        tps = avg_tokens_per_request / avg_latency_sec
                        print(f"吞吐量 (TPS, 估算): {tps:.2f} tokens/秒")
                print(f"总生成 tokens: {total_tokens}")

                # TPOT (Time Per Output Token) 计算
                # 对于每个请求，计算生成时间 / token 数
                tpot_values = []
                for r in successful:
                    if r.total_tokens > 0:
                        # 使用总延迟时间除以 token 数
                        tpot_ms = r.latency_ms / r.total_tokens
                        tpot_values.append(tpot_ms)

                if tpot_values:
                    print(f"\nTPOT (Time Per Output Token):")
                    print(f"  平均: {statistics.mean(tpot_values):.2f} ms/token")
                    print(f"  中位数: {statistics.median(tpot_values):.2f} ms/token")
                    print(f"  最小值: {min(tpot_values):.2f} ms/token")
                    print(f"  最大值: {max(tpot_values):.2f} ms/token")

            # 流式请求的首 token 延迟
            first_token_latencies = [
                r.first_token_ms for r in successful if r.first_token_ms is not None
            ]
            if first_token_latencies:
                print(f"\n首 Token 延迟 (ms) - 流式请求:")
                print(f"  平均: {statistics.mean(first_token_latencies):.2f}")
                print(f"  中位数: {statistics.median(first_token_latencies):.2f}")
                print(f"  最小值: {min(first_token_latencies):.2f}")
                print(f"  最大值: {max(first_token_latencies):.2f}")

        print("=" * 60 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="NanoRoute 字符串评测脚本")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:3001",
        help="NanoRoute 服务器地址 (默认: http://127.0.0.1:3001)",
    )
    parser.add_argument(
        "--model", type=str, default="opt-1.3b", help="模型名称 (默认: opt-1.3b)"
    )
    parser.add_argument(
        "--messages",
        type=str,
        default='[{"role": "user", "content": "Hello, how are you?"}]',
        help="消息列表 JSON 字符串",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=16, help="最大生成 token 数 (默认: 16)"
    )
    parser.add_argument(
        "--num-requests", type=int, default=10, help="请求总数 (默认: 10)"
    )
    parser.add_argument("--concurrency", type=int, default=1, help="并发数 (默认: 1)")
    parser.add_argument("--stream", action="store_true", help="使用流式请求")
    parser.add_argument("--warmup", type=int, default=0, help="预热请求数 (默认: 0)")

    args = parser.parse_args()

    # 解析消息
    try:
        messages = json.loads(args.messages)
        if not isinstance(messages, list):
            raise ValueError("messages 必须是列表")
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析 messages JSON: {e}")
        return

    benchmark = NanoRouteBenchmark(base_url=args.url)

    # 健康检查
    print("检查服务器健康状态...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.url}/health") as response:
                if response.status == 200:
                    print("✓ 服务器健康检查通过\n")
                else:
                    print(f"⚠ 警告: 健康检查返回状态码 {response.status}\n")
    except Exception as e:
        print(f"⚠ 警告: 无法连接到服务器: {e}\n")

    # 预热
    if args.warmup > 0:
        print(f"预热中 ({args.warmup} 个请求)...")
        await benchmark.concurrent_benchmark(
            messages=messages,
            num_requests=args.warmup,
            concurrency=1,
            model=args.model,
            max_tokens=args.max_tokens,
            stream=args.stream,
        )
        print("预热完成\n")

    # 执行评测
    print(f"开始性能评测...")
    print(f"  请求总数: {args.num_requests}")
    print(f"  并发数: {args.concurrency}")
    print(f"  流式模式: {'是' if args.stream else '否'}")
    print(f"  模型: {args.model}")
    print(f"  最大 tokens: {args.max_tokens}")
    print()

    start_time = time.time()
    results = await benchmark.concurrent_benchmark(
        messages=messages,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        model=args.model,
        max_tokens=args.max_tokens,
        stream=args.stream,
    )
    total_time = time.time() - start_time

    # 打印统计信息
    benchmark.print_statistics(results, total_time_seconds=total_time)
    print(f"总耗时: {total_time:.2f} 秒")


if __name__ == "__main__":
    asyncio.run(main())
