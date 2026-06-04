from locust import HttpUser, task, between, events
import json
import random


class InferenceUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"
        self.chat_payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is machine learning?"}
            ],
            "temperature": 0.7,
            "max_tokens": 256
        }
        self.headers = {"Authorization": "Bearer test-api-key"}

    @task(10)
    def chat_completion(self):
        with self.client.post(
            "/v1/chat/completions",
            json=self.chat_payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(5)
    def completion(self):
        payload = {
            "model": self.model_name,
            "prompt": "Once upon a time",
            "max_tokens": 100,
            "temperature": 0.7
        }

        with self.client.post(
            "/v1/completions",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(3)
    def embeddings(self):
        payload = {
            "model": self.model_name,
            "input": "Hello, world!"
        }

        with self.client.post(
            "/v1/embeddings",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def health_check(self):
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class StreamingUser(HttpUser):
    wait_time = between(2, 5)

    def on_start(self):
        self.headers = {"Authorization": "Bearer test-api-key"}

    @task
    def chat_completion_stream(self):
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [
                {"role": "user", "content": "Tell me a joke"}
            ],
            "max_tokens": 100,
            "stream": True
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=self.headers,
            stream=True,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                chunks = 0
                for line in response.iter_lines():
                    if line:
                        chunks += 1
                if chunks > 0:
                    response.success()
                else:
                    response.failure("No streaming chunks received")
            else:
                response.failure(f"HTTP {response.status_code}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("Load test starting...")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Load test completed")
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Total failures: {environment.stats.total.num_failures}")
    print(f"Average response time: {environment.stats.total.avg_response_time:.2f}ms")
    print(f"RPS: {environment.stats.total.total_rps:.2f}")
