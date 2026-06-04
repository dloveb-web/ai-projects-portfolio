import asyncio
import time
import statistics
import os
from typing import List, Dict, Any
import httpx


class BenchmarkRunner:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []
        self.api_key = os.getenv("INFERENCE_API_KEY", "test-api-key")

    async def run_chat_completion_benchmark(
        self,
        num_requests: int = 100,
        concurrency: int = 10,
        prompt: str = "What is artificial intelligence?"
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 256,
                "temperature": 0.7
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            latencies: List[float] = []
            successes = 0
            failures = 0

            async def single_request():
                nonlocal successes, failures
                start = time.time()
                try:
                    response = await client.post(
                        f"{self.base_url}/v1/chat/completions",
                        json=payload,
                        headers=headers
                    )
                    latency = time.time() - start

                    if response.status_code == 200:
                        latencies.append(latency)
                        successes += 1
                        return True
                    else:
                        failures += 1
                        return False
                except Exception as e:
                    failures += 1
                    return False

            semaphore = asyncio.Semaphore(concurrency)

            async def limited_request():
                async with semaphore:
                    return await single_request()

            tasks = [limited_request() for _ in range(num_requests)]
            await asyncio.gather(*tasks)

            if latencies:
                latencies.sort()
                return {
                    "total_requests": num_requests,
                    "successful_requests": successes,
                    "failed_requests": failures,
                    "success_rate": successes / num_requests * 100,
                    "latency_avg": statistics.mean(latencies),
                    "latency_median": statistics.median(latencies),
                    "latency_p50": latencies[int(len(latencies) * 0.50)],
                    "latency_p95": latencies[int(len(latencies) * 0.95)],
                    "latency_p99": latencies[int(len(latencies) * 0.99)],
                    "latency_min": min(latencies),
                    "latency_max": max(latencies),
                    "throughput": successes / sum(latencies) if latencies else 0
                }
            else:
                return {
                    "total_requests": num_requests,
                    "successful_requests": successes,
                    "failed_requests": failures,
                    "error": "All requests failed"
                }

    async def run_streaming_benchmark(
        self,
        num_requests: int = 50
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": [{"role": "user", "content": "Count to 10"}],
                "max_tokens": 100,
                "stream": True
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            start = time.time()
            total_chunks = 0
            successful_requests = 0

            async def single_stream_request():
                nonlocal total_chunks, successful_requests
                try:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/v1/chat/completions",
                        json=payload,
                        headers=headers
                    ) as response:
                        if response.status_code == 200:
                            chunk_count = 0
                            async for line in response.aiter_lines():
                                if line.startswith("data: "):
                                    chunk_count += 1
                            total_chunks += chunk_count
                            successful_requests += 1
                            return True
                        return False
                except Exception:
                    return False

            tasks = [single_stream_request() for _ in range(num_requests)]
            await asyncio.gather(*tasks)

            total_time = time.time() - start

            return {
                "total_requests": num_requests,
                "successful_requests": successful_requests,
                "total_time": total_time,
                "streaming_chunks": total_chunks,
                "avg_chunks_per_request": total_chunks / successful_requests if successful_requests > 0 else 0
            }

    async def run_health_check(self) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "response": response.json()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "status_code": response.status_code
                    }
            except Exception as e:
                return {
                    "status": "unreachable",
                    "error": str(e)
                }


async def main():
    runner = BenchmarkRunner()

    print("Running health check...")
    health = await runner.run_health_check()
    print(f"Health: {health}")

    print("\nRunning chat completion benchmark...")
    benchmark = await runner.run_chat_completion_benchmark(
        num_requests=100,
        concurrency=10
    )
    print(f"\nBenchmark Results:")
    print(f"  Total Requests: {benchmark['total_requests']}")
    print(f"  Success Rate: {benchmark.get('success_rate', 0):.2f}%")
    print(f"  Average Latency: {benchmark.get('latency_avg', 0):.3f}s")
    print(f"  P95 Latency: {benchmark.get('latency_p95', 0):.3f}s")
    print(f"  Throughput: {benchmark.get('throughput', 0):.2f} req/s")


if __name__ == "__main__":
    asyncio.run(main())
