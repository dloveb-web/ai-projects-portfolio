import pytest
from src.core.monitor import MetricsCollector, PrometheusMetrics


def test_metrics_collector_basic():
    collector = MetricsCollector()

    collector.record_request(0.1, success=True)
    collector.record_request(0.2, success=True)
    collector.record_request(0.3, success=True)
    collector.record_request(0.05, success=False)

    metrics = collector.get_metrics()

    assert metrics["total_requests"] == 4
    assert metrics["successful_requests"] == 3
    assert metrics["failed_requests"] == 1
    assert metrics["success_rate"] == 75.0


def test_metrics_collector_latency():
    collector = MetricsCollector()

    for i in range(100):
        collector.record_request(0.1 * (i + 1), success=True)

    metrics = collector.get_metrics()

    assert metrics["p50_latency"] > 0
    assert metrics["p95_latency"] > metrics["p50_latency"]
    assert metrics["p99_latency"] > metrics["p95_latency"]


def test_endpoint_metrics():
    collector = MetricsCollector()

    collector.record_endpoint_request("/v1/chat/completions", 0.1, success=True)
    collector.record_endpoint_request("/v1/chat/completions", 0.2, success=True)
    collector.record_endpoint_request("/v1/embeddings", 0.05, success=True)

    metrics = collector.get_metrics()

    assert "/v1/chat/completions" in metrics["endpoint_metrics"]
    assert metrics["endpoint_metrics"]["/v1/chat/completions"]["count"] == 2
    assert metrics["endpoint_metrics"]["/v1/embeddings"]["count"] == 1


def test_prometheus_metrics():
    prom = PrometheusMetrics()

    prom.increment("requests_total")
    prom.increment("requests_success")
    prom.record_latency(0.1)
    prom.record_latency(0.2)
    prom.record_tokens(100)

    metrics = prom.get_all_metrics()

    assert metrics["requests_total"] == 1
    assert metrics["requests_success"] == 1
    assert metrics["latency_avg"] == 0.15
    assert metrics["tokens_total"] == 100


def test_prometheus_format():
    prom = PrometheusMetrics()

    prom.increment("requests_total", 10)

    output = prom.format_prometheus()

    assert "inference_requests_total 10" in output
    assert "# TYPE inference_requests_total counter" in output
