#!/usr/bin/env python3
"""
Simple test script to debug the LLM initialization issue.
"""
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

print("=== Environment Check ===")
print(f"DEMO_MODE: '{os.getenv('DEMO_MODE')}'")
print(f"GOOGLE_API_KEY: '{os.getenv('GOOGLE_API_KEY')[:20]}...'")
print(f"ANTHROPIC_API_KEY: '{os.getenv('ANTHROPIC_API_KEY')[:20]}...'")
print(f"OPENAI_API_KEY: '{os.getenv('OPENAI_API_KEY')[:20]}...'")

print("\n=== Testing get_llm function ===")
try:
    from utils.helpers import get_llm
    llm = get_llm()
    print(f"SUCCESS: LLM initialized with {type(llm).__name__}")
    if hasattr(llm, 'model_name'):
        print(f"Model: {llm.model_name}")
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc()