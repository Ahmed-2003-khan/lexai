import os
import sys
import json
import asyncio
import httpx
from typing import List
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase
from test_cases import EVAL_TEST_CASES

# Define the base URL for the API and the directory to store results
API_BASE_URL = "http://localhost:8000/api/v1"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

async def authenticate() -> str:
    """Authenticates with the API and retrieves a JWT token. Automatically registers if needed."""
    auth_data = {"username": "ahmed_admin", "password": "supersecretpassword"}
    reg_data = {"username": "ahmed_admin", "email": "admin@lexai.com", "password": "supersecretpassword"}
    
    # Initialize the HTTP client with a 60-second timeout to accommodate initial server load delays
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Attempt to register the user first in case the database is empty
        await client.post(f"{API_BASE_URL}/auth/register", json=reg_data)
        
        # Proceed to login and retrieve the token
        response = await client.post(f"{API_BASE_URL}/auth/token", data=auth_data)
        if response.status_code == 200:
            return response.json().get("access_token")
        raise RuntimeError(f"Failed to authenticate. API Response: {response.text}")

async def fetch_api_response(query: str, token: str) -> dict:
    """Calls the live LexAI API to retrieve the answer and context."""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"query": query, "jurisdiction": "Pakistan"}
    
    # Initialize the HTTP client with a 180-second timeout to allow the agentic workflow enough time to process
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(f"{API_BASE_URL}/query", json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()
        return {"answer": "API Error", "citations": []}

def format_metric_result(metric, threshold: float) -> str:
    """Formats the score with a pass/fail emoji."""
    if not hasattr(metric, 'score') or metric.score is None:
        return "⚠️ N/A"
    
    passed = metric.score >= threshold if metric.__class__.__name__ != "HallucinationMetric" else metric.score <= threshold
    icon = "✅" if passed else "❌"
    return f"{icon} {metric.score:.2f}"

async def main():
    """Executes the evaluation pipeline against a minimal subset of test cases using a cost-effective model."""
    print("Starting LexAI DeepEval Pipeline (Cost-Saving Mode: 1 Case, GPT-4o-Mini)...")
    token = await authenticate()
    
    # Define the evaluation model to control costs during metric computation
    eval_model = "gpt-4o-mini"
    
    # Initialize metrics with specific thresholds and the designated evaluation model
    relevancy = AnswerRelevancyMetric(threshold=0.7, model=eval_model)
    faithfulness = FaithfulnessMetric(threshold=0.75, model=eval_model)
    c_precision = ContextualPrecisionMetric(threshold=0.6, model=eval_model)
    c_recall = ContextualRecallMetric(threshold=0.6, model=eval_model)
    hallucination = HallucinationMetric(threshold=0.25, model=eval_model)
    
    metrics = [relevancy, faithfulness, c_precision, c_recall, hallucination]
    results_log = []
    tests_passed = 0
    
    # Restrict the evaluation to a single test case to minimize API usage
    test_subset = EVAL_TEST_CASES[:1]
    
    print("\n┌" + "─"*32 + "┬" + "─"*10 + "┬" + "─"*13 + "┬" + "─"*11 + "┬" + "─"*10 + "┬" + "─"*14 + "┐")
    print("│ Test Case                      │ Relevancy│ Faithfulness│ C.Precision│ C.Recall │ Hallucination│")
    print("├" + "─"*32 + "┼" + "─"*10 + "┼" + "─"*13 + "┼" + "─"*11 + "┼" + "─"*10 + "┼" + "─"*14 + "┤")

    for case in test_subset:
        api_data = await fetch_api_response(case["input"], token)
        
        actual_output = api_data.get("answer", "")
        
        raw_citations = api_data.get("citations", [])
        retrieval_context = []
        for c in raw_citations:
            if isinstance(c, dict):
                retrieval_context.append(c.get("content_snippet", str(c)))
            elif isinstance(c, str):
                retrieval_context.append(c)
                
        if not retrieval_context:
            retrieval_context = ["No context retrieved."]
            
        test_case = LLMTestCase(
            input=case["input"],
            actual_output=actual_output,
            expected_output=case["expected_output"],
            retrieval_context=retrieval_context,
            context=retrieval_context 
        )
        
        case_passed = True
        scores = {}
        
        for metric in metrics:
            await metric.a_measure(test_case)
            scores[metric.__class__.__name__] = metric.score
            if not metric.is_successful():
                case_passed = False
                
        if case_passed:
            tests_passed += 1

        name_trunc = (case["id"][:29] + '...') if len(case["id"]) > 29 else case["id"].ljust(30)
        print(f"│ {name_trunc} │ {format_metric_result(relevancy, 0.7):<8} │ {format_metric_result(faithfulness, 0.75):<11} │ {format_metric_result(c_precision, 0.6):<9} │ {format_metric_result(c_recall, 0.6):<8} │ {format_metric_result(hallucination, 0.25):<12} │")
        
        results_log.append({
            "test_id": case["id"],
            "input": case["input"],
            "scores": scores,
            "passed": case_passed
        })

    print("└" + "─"*32 + "┴" + "─"*10 + "┴" + "─"*13 + "┴" + "─"*11 + "┴" + "─"*10 + "┴" + "─"*14 + "┘")
    
    pass_rate = (tests_passed / len(test_subset)) * 100
    print(f"\nOverall pass rate: {tests_passed}/{len(test_subset)} ({pass_rate:.0f}%) {'✅' if pass_rate >= 70 else '❌'}")

if __name__ == "__main__":
    asyncio.run(main())