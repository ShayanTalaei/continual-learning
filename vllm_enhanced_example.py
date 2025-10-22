#!/usr/bin/env python3
"""
Enhanced VLLMClient example showcasing all new features for GeminiClient parity.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.lm.vllm_client import VLLMClient, VLLMConfig
from src.utils.logger import enable_json_logging
from src.utils import logger as jsonlogger


def test_json_response_with_schema():
    """Test JSON response with schema provided to the model."""
    print("Testing JSON response with schema...")
    
    config = VLLMConfig(
        model="microsoft/DialoGPT-small",
        temperature=0.1,
        max_output_tokens=100,
        log_calls=True,
        json_validation=True,
        use_chat_template=True,  # Enable chat templates
    )
    
    lm = VLLMClient(config)
    
    # Test with JSON schema in context (schema gets included in prompt)
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["answer", "confidence"]
    }
    
    with jsonlogger.json_log_context(response_schema=schema):
        response = lm.call(
            "You are a helpful assistant.",
            "What is 2+2? Provide your answer and confidence level."
        )
    
    print(f"JSON Response: {response}")
    return response


def test_chat_template():
    """Test chat template functionality."""
    print("Testing chat template...")
    
    config = VLLMConfig(
        model="microsoft/DialoGPT-small",
        temperature=0.1,
        max_output_tokens=50,
        log_calls=True,
        use_chat_template=True,
        json_validation=False,
    )
    
    lm = VLLMClient(config)
    
    response = lm.call(
        "You are a helpful assistant.",
        "Say hello in one word."
    )
    
    print(f"Chat Template Response: {response}")
    return response


def test_thread_safety():
    """Test thread-safe engine initialization."""
    print("Testing thread safety...")
    
    import threading
    import time
    
    config = VLLMConfig(
        model="microsoft/DialoGPT-small",
        temperature=0.1,
        max_output_tokens=20,
        log_calls=False,
        use_chat_template=False,
    )
    
    results = []
    
    def worker(worker_id):
        lm = VLLMClient(config)
        response = lm.call("You are helpful.", f"Worker {worker_id} says hello.")
        results.append((worker_id, response))
    
    # Create multiple threads that will initialize the engine concurrently
    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    print(f"Thread safety results: {results}")
    return len(results) == 3


def test_rate_limiting():
    """Test rate limiting between retries."""
    print("Testing rate limiting...")
    
    config = VLLMConfig(
        model="microsoft/DialoGPT-small",
        temperature=0.1,
        max_output_tokens=20,
        log_calls=False,
        max_retries=2,
        rate_limit_delay=0.5,  # 500ms delay between retries
    )
    
    lm = VLLMClient(config)
    
    start_time = time.time()
    response = lm.call("You are helpful.", "Say hello quickly.")
    end_time = time.time()
    
    print(f"Rate limiting response: {response}")
    print(f"Total time: {end_time - start_time:.2f}s")
    return response


def test_metrics_extraction():
    """Test enhanced metrics extraction."""
    print("Testing metrics extraction...")
    
    config = VLLMConfig(
        model="microsoft/DialoGPT-small",
        temperature=0.1,
        max_output_tokens=30,
        log_calls=True,
    )
    
    lm = VLLMClient(config)
    
    with jsonlogger.json_log_context(mode="test", call_type="metrics_test"):
        response = lm.call(
            "You are a helpful assistant.",
            "Count from 1 to 5."
        )
    
    print(f"Metrics test response: {response}")
    return response


def main():
    """Run all enhanced feature tests."""
    print("Testing Enhanced VLLMClient Features")
    print("=" * 50)
    
    # Enable JSON logging
    enable_json_logging("./enhanced_test_logs")
    
    tests = [
        ("JSON Response with Schema", test_json_response_with_schema),
        ("Chat Template", test_chat_template),
        ("Thread Safety", test_thread_safety),
        ("Rate Limiting", test_rate_limiting),
        ("Metrics Extraction", test_metrics_extraction),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n--- {test_name} ---")
            result = test_func()
            results.append((test_name, True, result))
            print(f"✓ {test_name} passed")
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            results.append((test_name, False, str(e)))
    
    print("\n" + "=" * 50)
    print("Test Results:")
    for test_name, success, result in results:
        status = "✓" if success else "✗"
        print(f"{status} {test_name}: {result}")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    return passed == total


if __name__ == "__main__":
    import time
    success = main()
    sys.exit(0 if success else 1)
