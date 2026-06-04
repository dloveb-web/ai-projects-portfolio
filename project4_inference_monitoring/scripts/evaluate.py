import asyncio
import httpx
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


class EvaluationRunner:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_cases: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []
        self.api_key = os.getenv("INFERENCE_API_KEY", "test-api-key")

    def load_test_cases(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.test_cases = json.load(f)
        print(f"Loaded {len(self.test_cases)} test cases")

    async def chat_completion(self, prompt: str, context: Optional[str] = None) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            messages = []
            if context:
                messages.append({"role": "system", "content": f"Context: {context}"})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": messages,
                "max_tokens": 256,
                "temperature": 0.7
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            try:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers
                )
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"HTTP {response.status_code}")
            except Exception as e:
                print(f"Error: {e}")
                return ""

    async def evaluate_ragas_metrics(
        self,
        questions: List[str],
        contexts: List[str],
        ground_truths: List[str]
    ) -> Dict[str, Any]:
        responses = []
        for question in questions:
            response = await self.chat_completion(question)
            responses.append(response)

        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            )
            from datasets import Dataset

            data = {
                "question": questions,
                "answer": responses,
                "contexts": [[ctx] for ctx in contexts],
                "ground_truth": ground_truths
            }

            dataset = Dataset.from_dict(data)
            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            result = evaluate(dataset, metrics)

            return {
                "ragas_score": result["ragas_score"],
                "faithfulness": result["faithfulness"],
                "answer_relevancy": result["answer_relevancy"],
                "context_precision": result["context_precision"],
                "context_recall": result["context_recall"],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"RAGAS evaluation error: {e}")
            return {
                "error": str(e),
                "manual_evaluation": True,
                "responses": responses
            }

    async def run_evaluation(self) -> Dict[str, Any]:
        print("Starting evaluation...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "total_cases": len(self.test_cases),
            "passed": 0,
            "failed": 0,
            "details": []
        }

        for i, test_case in enumerate(self.test_cases):
            question = test_case.get("question", "")
            expected = test_case.get("expected", "")
            category = test_case.get("category", "unknown")

            response = await self.chat_completion(question)

            passed = self._check_response(response, expected)

            results["details"].append({
                "case_id": i + 1,
                "question": question,
                "response": response,
                "expected": expected,
                "passed": passed,
                "category": category
            })

            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1

            print(f"Case {i+1}/{len(self.test_cases)}: {'PASS' if passed else 'FAIL'}")

        results["pass_rate"] = results["passed"] / results["total_cases"] * 100

        self.results = results
        return results

    def _check_response(self, response: str, expected: str) -> bool:
        response_lower = response.lower()
        expected_keywords = expected.lower().split()

        matches = sum(1 for keyword in expected_keywords if keyword in response_lower)

        return matches >= len(expected_keywords) * 0.5

    def save_results(self, filepath: str = "evaluation_results.json"):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filepath}")


async def main():
    runner = EvaluationRunner()

    runner.test_cases = [
        {
            "question": "What is machine learning?",
            "expected": "algorithm data learn",
            "category": "definition"
        },
        {
            "question": "Explain neural networks",
            "expected": "neuron connection layer",
            "category": "technical"
        },
        {
            "question": "What is Python?",
            "expected": "programming language",
            "category": "definition"
        }
    ]

    results = await runner.run_evaluation()

    print(f"\nEvaluation Summary:")
    print(f"  Total: {results['total_cases']}")
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Pass Rate: {results['pass_rate']:.2f}%")

    runner.save_results()


if __name__ == "__main__":
    asyncio.run(main())
