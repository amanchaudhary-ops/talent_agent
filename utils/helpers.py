"""
utils/helpers.py

Shared utility functions for score calculation, JSON parsing, and formatting.
"""

import json
import re
from typing import Any


class MockLLM:
    """Mock LLM for demo mode - returns realistic responses without API calls."""
    
    def __init__(self):
        self.temperature = 0.1
        self.model_name = "mock-llm-demo"
    
    def invoke(self, prompt: str) -> Any:
        """Return mock responses based on prompt content."""
        if isinstance(prompt, list):
            # Handle message list format
            prompt_text = str(prompt)
        else:
            prompt_text = str(prompt)
            
        if "role_title" in prompt_text.lower() or "required_skills" in prompt_text.lower():
            # JD parsing response
            return type('MockResponse', (), {
                'content': '''{
                    "role_title": "Senior Python Developer",
                    "required_skills": ["Python", "Django", "PostgreSQL", "AWS"],
                    "preferred_skills": ["Docker", "Kubernetes", "React"],
                    "seniority_level": "senior",
                    "role_type": "full-time",
                    "location": "remote",
                    "compensation_range": "$120,000 - $160,000",
                    "must_haves": ["5+ years Python experience", "Web framework experience"],
                    "nice_to_haves": ["Cloud experience", "DevOps knowledge"],
                    "summary": "Senior Python Developer role focusing on web applications and cloud infrastructure."
                }'''
            })()
        elif "candidate" in prompt_text.lower() or "score" in prompt_text.lower():
            # Candidate scoring response
            return type('MockResponse', (), {
                'content': '{"match_score": 85, "match_reasons": ["Strong Python experience", "Django expertise"], "match_gaps": ["Limited AWS experience"]}'
            })()
        else:
            # Generic response
            return type('MockResponse', (), {
                'content': 'This is a mock response for demo purposes. Add a real API key to get actual AI responses.'
            })()
    
    def with_structured_output(self, schema):
        """Mock structured output - just return self."""
        return self


def safe_parse_json(text: str) -> dict:
    """
    Safely parse a JSON string from an LLM response.
    Strips markdown code fences if present.

    Args:
        text: Raw LLM output string.

    Returns:
        Parsed dict, or empty dict on failure.
    """
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def compute_composite_score(match_score: float, interest_score: float, alpha: float, beta: float) -> float:
    """
    Compute the weighted composite score from match and interest scores.

    Args:
        match_score: Candidate match score (0-100).
        interest_score: Candidate interest score (0-100).
        alpha: Weight applied to match_score.
        beta: Weight applied to interest_score.

    Returns:
        Composite score rounded to 2 decimal places.
    """
    return round(alpha * match_score + beta * interest_score, 2)


import os


def get_llm(temperature: float = 0.1) -> Any:
    """
    Get an LLM instance based on available API keys.
    Prioritizes: Anthropic (Claude) > OpenAI (GPT) > Google (Gemini).
    
    All providers now support the required functionality through fallback methods.
    
    Args:
        temperature: Sampling temperature for the LLM.
    
    Returns:
        Configured LLM instance.
    
    Raises:
        ValueError: If no supported API key is found or all fail.
    """
    
    # Check for demo mode first
    demo_mode = os.getenv("DEMO_MODE", "").lower()
    print(f"DEBUG: DEMO_MODE = '{demo_mode}'")
    if demo_mode in ["true", "1", "yes"]:
        print("🎭 DEMO MODE: Using mock LLM for interface testing")
        return MockLLM()
    
    # Try Anthropic Claude (best support for structured output)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key and anthropic_key not in ["your_claude_key", "sk-ant-..."]:
        try:
            from langchain_anthropic import ChatAnthropic
            print("✓ Using Anthropic Claude API")
            return ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                api_key=anthropic_key,
                temperature=temperature,
                timeout=60,
            )
        except Exception as e:
            print(f"✗ Anthropic API error: {str(e)[:100]}")
    
    # Try OpenAI GPT-4o (good support for structured output)
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and openai_key not in ["your_openai_key", "sk-..."]:
        try:
            from langchain_openai import ChatOpenAI
            print("✓ Using OpenAI GPT API")
            return ChatOpenAI(
                model="gpt-4o",
                api_key=openai_key,
                temperature=temperature,
                timeout=60,
            )
        except Exception as e:
            print(f"✗ OpenAI API error: {str(e)[:100]}")
    
    # Google Gemini (Fully supported with JSON fallback)
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    print(f"DEBUG: GOOGLE_API_KEY loaded: '{google_key[:10]}...' (length: {len(google_key)})")
    if google_key and google_key not in ["your_google_key"]:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print("✓ Using Google Gemini API")
            print(f"DEBUG: Initializing with model='gemini-2.5-flash', key starts with: {google_key[:10]}")
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=google_key,
                temperature=temperature,
            )
            
            # Skip API key test if SKIP_API_TEST is set
            skip_test = os.getenv("SKIP_API_TEST", "").lower() in ["true", "1", "yes"]
            if skip_test:
                print("⚠️ SKIPPING API key validation (use with caution)")
                return llm
            
            # Test the LLM with a simple call to verify it works
            try:
                print("DEBUG: Testing API key with simple call...")
                test_response = llm.invoke("Say 'OK' if you can read this.")
                print(f"DEBUG: Test response type: {type(test_response)}")
                if hasattr(test_response, 'content'):
                    print(f"DEBUG: Response content: '{test_response.content[:50]}...'")
                else:
                    print(f"DEBUG: Response attributes: {dir(test_response)}")
                
                if not test_response or not hasattr(test_response, 'content') or not test_response.content:
                    raise ValueError("Gemini API returned empty or invalid response")
                print("✓ Gemini API key validated successfully")
            except Exception as test_e:
                error_str = str(test_e).lower()
                print(f"DEBUG: Full error: {str(test_e)}")
                if "api_key" in error_str or "invalid" in error_str or "unauthorized" in error_str:
                    print(f"✗ INVALID API KEY: {str(test_e)}")
                    print("   → Your Gemini API key appears to be invalid or expired")
                    print("   → Get a new key from: https://ai.google.dev/")
                elif "quota" in error_str or "rate" in error_str or "limit" in error_str:
                    print(f"✗ QUOTA EXCEEDED: {str(test_e)}")
                    print("   → You've exceeded your Gemini API quota")
                    print("   → Check your usage at: https://ai.google.dev/")
                elif "model" in error_str or "not found" in error_str:
                    print(f"✗ MODEL ERROR: {str(test_e)}")
                    print("   → The specified model may not be available")
                    print("   → Try a different model or check availability")
                else:
                    print(f"✗ API ERROR: {str(test_e)}")
                    print("   → Unexpected error occurred")
                raise  # Re-raise to trigger the error handling below
            return llm
        except Exception as e:
            error_msg = str(e).lower()
            if "api_key" in error_msg or "invalid" in error_msg or "unauthorized" in error_msg:
                print(f"✗ Gemini API key invalid: {str(e)[:100]}")
                print("   → Get a new key from: https://ai.google.dev/")
            elif "quota" in error_msg or "rate" in error_msg:
                print(f"✗ Gemini API quota exceeded: {str(e)[:100]}")
            elif "model" in error_msg:
                print(f"✗ Gemini model not available: {str(e)[:100]}")
            else:
                print(f"✗ Gemini API error: {str(e)[:100]}")
    
    # If we get here, no valid API key was found
    error_msg = (
        "\n" + "="*60 + "\n"
        "❌ NO VALID API KEY FOUND\n"
        "="*60 + "\n\n"
        "Choose ONE option below:\n\n"
        "🔥 RECOMMENDED: Claude (best quality)\n"
        "   1. Go to: https://console.anthropic.com/\n"
        "   2. Create account & get API key\n"
        "   3. Add to .env: ANTHROPIC_API_KEY=sk-ant-v1-your-key-here\n\n"
        "💰 OpenAI GPT-4o (reliable)\n"
        "   1. Go to: https://platform.openai.com/api-keys\n"
        "   2. Create API key\n"
        "   3. Add to .env: OPENAI_API_KEY=sk-proj-your-key-here\n\n"
        "🆓 Google Gemini (free but less reliable)\n"
        "   1. Go to: https://ai.google.dev/\n"
        "   2. Click 'Get API Key'\n"
        "   3. Add to .env: GOOGLE_API_KEY=your-actual-key-here\n\n"
        "After updating .env, restart the app.\n"
        "="*60 + "\n\n"
        "🚀 DEMO MODE: Want to see the app interface?\n"
        "   Set DEMO_MODE=true in .env to run with mock data\n"
        "="*60
    )
    raise ValueError(error_msg)


def determine_recommended_action(composite_score: float) -> str:
    """
    Determine the recruiter action based on composite score.

    Args:
        composite_score: The weighted combined score.

    Returns:
        One of: 'Schedule interview', 'Follow up', 'Pass'.
    """
    if composite_score >= 75:
        return "Schedule interview"
    elif composite_score >= 55:
        return "Follow up"
    else:
        return "Pass"


def format_transcript_for_prompt(transcript: list[dict]) -> str:
    """
    Format a conversation transcript list into a readable string for prompts.

    Args:
        transcript: List of {role, content} dicts.

    Returns:
        Multi-line string of the conversation.
    """
    lines = []
    for turn in transcript:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def truncate_list(items: list[Any], max_items: int = 3) -> list[Any]:
    """Return at most max_items from a list."""
    return items[:max_items]