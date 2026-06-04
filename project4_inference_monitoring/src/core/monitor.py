import time
from typing import Dict, Any
from collections import defaultdict, deque
import threading

try:
    from prometheus_client import Histogram, Counter, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricsCollector:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.request_latencies = deque(maxlen=window_size)
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.endpoint_metrics = defaultdict(lambda: {
            "count": 0,
            "errors": 0,
            "latencies": deque(maxlen=100)
        })

    def record_request(self, latency: float, success: bool = True, tokens: int = 0):
        with self.lock:
            self.request_latencies.append(latency)
            self.total_requests += 1

            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

            self.total_tokens += tokens

    def record_endpoint_request(self, endpoint: str, latency: float, success: bool = True):
        with self.lock:
            metrics = self.endpoint_metrics[endpoint]
            metrics["count"] += 1
            metrics["latencies"].append(latency)

            if not success:
                metrics["errors"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self.lock:
            uptime = time.time() - self.start_time

            if self.request_latencies:
                sorted_latencies = sorted(self.request_latencies)
                p50_idx = int(len(sorted_latencies) * 0.50)
                p95_idx = int(len(sorted_latencies) * 0.95)
                p99_idx = int(len(sorted_latencies) * 0.99)

                p50_latency = sorted_latencies[p50_idx] if sorted_latencies else 0
                p95_latency = sorted_latencies[p95_idx] if sorted_latencies else 0
                p99_latency = sorted_latencies[p99_idx] if sorted_latencies else 0
                avg_latency = sum(sorted_latencies) / len(sorted_latencies)
            else:
                p50_latency = p95_latency = p99_latency = avg_latency = 0

            success_rate = (
                self.successful_requests / self.total_requests * 100
                if self.total_requests > 0 else 0
            )

            requests_per_second = self.total_requests / uptime if uptime > 0 else 0

            tokens_per_second = self.total_tokens / uptime if uptime > 0 else 0

            return {
                "uptime_seconds": uptime,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
                "average_latency": avg_latency,
                "p50_latency": p50_latency,
                "p95_latency": p95_latency,
                "p99_latency": p99_latency,
                "requests_per_second": requests_per_second,
                "total_tokens": self.total_tokens,
                "tokens_per_second": tokens_per_second,
                "endpoint_metrics": self._get_endpoint_metrics()
            }

    def _get_endpoint_metrics(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for endpoint, metrics in self.endpoint_metrics.items():
            latencies = list(metrics["latencies"])
            if latencies:
                result[endpoint] = {
                    "count": metrics["count"],
                    "errors": metrics["errors"],
                    "error_rate": metrics["errors"] / metrics["count"] * 100 if metrics["count"] > 0 else 0,
                    "avg_latency": sum(latencies) / len(latencies),
                    "min_latency": min(latencies),
                    "max_latency": max(latencies)
                }
            else:
                result[endpoint] = {
                    "count": metrics["count"],
                    "errors": metrics["errors"],
                    "error_rate": 0,
                    "avg_latency": 0,
                    "min_latency": 0,
                    "max_latency": 0
                }
        return result

    def reset(self):
        with self.lock:
            self.request_latencies.clear()
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.total_tokens = 0
            self.start_time = time.time()
            self.endpoint_metrics.clear()


class PrometheusMetrics:
    def __init__(self):
        if PROMETHEUS_AVAILABLE:
            self.latency_histogram = Histogram(
                'inference_latency_seconds',
                'Inference request latency',
                ['endpoint'],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
            )
            self.requests_counter = Counter(
                'inference_requests_total',
                'Total number of inference requests',
                ['endpoint', 'status']
            )
            self.tokens_counter = Counter(
                'inference_tokens_total',
                'Total tokens processed',
                ['endpoint']
            )
            self.in_flight_gauge = Gauge(
                'inference_in_flight_requests',
                'Number of requests in flight'
            )

        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "latency_sum": 0.0,
            "latency_count": 0,
            "tokens_total": 0,
            "in_flight_requests": 0
        }
        self.lock = threading.Lock()

    def increment(self, metric: str, value: int = 1):
        with self.lock:
            if metric in self.metrics:
                self.metrics[metric] += value

    def decrement(self, metric: str, value: int = 1):
        with self.lock:
            if metric in self.metrics:
                self.metrics[metric] -= value

    def record_latency(self, latency: float, endpoint: str = "unknown"):
        if PROMETHEUS_AVAILABLE:
            self.latency_histogram.labels(endpoint=endpoint).observe(latency)

        with self.lock:
            self.metrics["latency_sum"] += latency
            self.metrics["latency_count"] += 1

    def record_request(self, endpoint: str = "unknown", success: bool = True):
        status = "success" if success else "failed"

        if PROMETHEUS_AVAILABLE:
            self.requests_counter.labels(endpoint=endpoint, status=status).inc()

        with self.lock:
            self.metrics["requests_total"] += 1
            if success:
                self.metrics["requests_success"] += 1
            else:
                self.metrics["requests_failed"] += 1

    def record_tokens(self, tokens: int, endpoint: str = "unknown"):
        if PROMETHEUS_AVAILABLE:
            self.tokens_counter.labels(endpoint=endpoint).inc(tokens)

        with self.lock:
            self.metrics["tokens_total"] += tokens

    def set_in_flight(self, value: int):
        if PROMETHEUS_AVAILABLE:
            self.in_flight_gauge.set(value)

        with self.lock:
            self.metrics["in_flight_requests"] = value

    def get_all_metrics(self) -> Dict[str, float]:
        with self.lock:
            result = self.metrics.copy()

            if result["latency_count"] > 0:
                result["latency_avg"] = result["latency_sum"] / result["latency_count"]
            else:
                result["latency_avg"] = 0.0

            return result

    def format_prometheus(self) -> str:
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import generate_latest
            return generate_latest().decode('utf-8')
        else:
            metrics = self.get_all_metrics()

            output = []
            output.append(f'# HELP inference_requests_total Total number of inference requests')
            output.append(f'# TYPE inference_requests_total counter')
            output.append(f'inference_requests_total {metrics["requests_total"]}')

            output.append(f'# HELP inference_requests_success Number of successful requests')
            output.append(f'# TYPE inference_requests_success counter')
            output.append(f'inference_requests_success {metrics["requests_success"]}')

            output.append(f'# HELP inference_requests_failed Number of failed requests')
            output.append(f'# TYPE inference_requests_failed counter')
            output.append(f'inference_requests_failed {metrics["requests_failed"]}')

            output.append(f'# HELP inference_latency_seconds Request latency in seconds')
            output.append(f'# TYPE inference_latency_seconds histogram')
            output.append(f'inference_latency_sum {metrics["latency_sum"]}')
            output.append(f'inference_latency_count {metrics["latency_count"]}')

            output.append(f'# HELP inference_tokens_total Total tokens processed')
            output.append(f'# TYPE inference_tokens_total counter')
            output.append(f'inference_tokens_total {metrics["tokens_total"]}')

            output.append(f'# HELP inference_in_flight_requests Number of requests in flight')
            output.append(f'# TYPE inference_in_flight_requests gauge')
            output.append(f'inference_in_flight_requests {metrics["in_flight_requests"]}')

            return "\n".join(output)


_global_metrics = MetricsCollector()
_global_prometheus = PrometheusMetrics()


def get_metrics_collector() -> MetricsCollector:
    return _global_metrics


def get_prometheus_metrics() -> PrometheusMetrics:
    return _global_prometheus
