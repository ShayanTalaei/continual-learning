#!/usr/bin/env python3
"""
Test script for vLLM integration with the continual learning framework.
This script tests the VLLMClient without requiring a full training run.
"""

import os
import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.lm.vllm_client import VLLMClient, VLLMConfig
from src.lm.lm_factory import get_lm_client
from src.utils.logger import enable_json_logging
from src.utils import logger as jsonlogger


def test_direct_vllm_client():
    """Test VLLMClient directly."""
    print("Testing VLLMClient directly...")
    
    # Enable JSON logging for this test
    enable_json_logging("./test_llm_calls")
    
    config = VLLMConfig(
        model="vllm:microsoft/DialoGPT-small",  # Test with vllm: prefix
        temperature=0.2,
        max_output_tokens=50,
        log_calls=True,
        max_retries=1,
        tensor_parallel_size=1,
        # Enhanced features
        use_chat_template=True,
        json_validation=True,
        rate_limit_delay=0.1,
    )
    
    try:
        lm = VLLMClient(config)
        
        # Test with logging context
        with jsonlogger.json_log_context(mode="test", episode_index=1, step_index=1, call_type="action"):
            response = lm.call(
                "You are a helpful assistant.",
                "Say hello in one word."
            )
        
        print(f"Response: {response}")
        print("✓ Direct VLLMClient test passed")
        return True
        
    except Exception as e:
        print(f"✗ Direct VLLMClient test failed: {e}")
        return False


def test_factory_routing():
    """Test that lm_factory correctly routes vllm: prefixed models."""
    print("Testing factory routing...")
    
    config = VLLMConfig(
        model="microsoft/DialoGPT-small",  # Test without prefix
        temperature=0.2,
        max_output_tokens=50,
        log_calls=False,  # Disable logging for this test
        max_retries=1,
    )
    
    try:
        lm = get_lm_client(config)
        assert isinstance(lm, VLLMClient), f"Expected VLLMClient, got {type(lm)}"
        
        response = lm.call(
            "You are a helpful assistant.",
            "Say hello."
        )
        
        print(f"Response: {response}")
        print("✓ Factory routing test passed")
        return True
        
    except Exception as e:
        print(f"✗ Factory routing test failed: {e}")
        return False


def test_config_coercion():
    """Test that LMConfig gets coerced to VLLMConfig when needed."""
    print("Testing config coercion...")
    
    from src.lm.language_model import LMConfig
    
    # Start with base LMConfig
    base_config = LMConfig(
        model="vllm:microsoft/DialoGPT-small",  # Test prefix routing
        temperature=0.2,
        max_output_tokens=50,
        log_calls=False,
        max_retries=1,
    )
    
    try:
        lm = get_lm_client(base_config)
        assert isinstance(lm, VLLMClient), f"Expected VLLMClient, got {type(lm)}"
        assert isinstance(lm.config, VLLMConfig), f"Expected VLLMConfig, got {type(lm.config)}"
        
        print("✓ Config coercion test passed")
        return True
        
    except Exception as e:
        print(f"✗ Config coercion test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing vLLM integration...")
    print("=" * 50)
    
    # Check if vllm is available
    try:
        import vllm
        print(f"vLLM version: {vllm.__version__}")
    except ImportError:
        print("✗ vLLM not installed. Install with: pip install vllm")
        return False
    
    tests = [
        test_direct_vllm_client,
        test_factory_routing,
        test_config_coercion,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
