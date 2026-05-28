"""Test dynamic system prompt generation."""
import sys, os, asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi_agent.agent import Agent
from pi_agent.llm import Model


def test_static_prompt_gets_auto_injected():
    """String system_prompt should get <env> and <tools> appended."""
    agent = Agent(
        model=Model(id="gpt-4o", provider="openai"),
        system_prompt="You are a test assistant.",
        cwd="/fake/project",
    )

    prompt = agent.system_prompt
    print(f"  System prompt length: {len(prompt)}")

    assert "You are a test assistant." in prompt, "Should contain static part"
    assert "<env>" in prompt, "Should contain <env> section"
    assert "</env>" in prompt
    assert "Working directory: " in prompt
    assert "fake" in prompt or "project" in prompt  # resolve() may alter path
    assert "Current date:" in prompt
    assert "Operating system:" in prompt

    print("Test 1 PASSED: static prompt auto-injected")


def test_prompt_with_tools():
    """<tools> section should list registered tools."""
    from pi_agent.tools import create_default_tools

    tools = create_default_tools(cwd="/fake/project")
    agent = Agent(
        model=Model(id="gpt-4o", provider="openai"),
        system_prompt="You are a test assistant.",
        tools=tools,
        cwd="/fake/project",
    )

    prompt = agent.system_prompt
    assert "<tools>" in prompt, "Should contain <tools> section"
    assert "</tools>" in prompt
    assert "- read:" in prompt, "Should list read tool"
    assert "- write:" in prompt, "Should list write tool"
    assert "- bash:" in prompt, "Should list bash tool"

    print("Test 2 PASSED: tools listed in prompt")


def test_callable_prompt():
    """Callable system_prompt gets context dict and full control."""
    captured_ctx = {}

    def custom_prompt(ctx: dict) -> str:
        nonlocal captured_ctx
        captured_ctx = ctx
        return f"Custom prompt for {ctx['cwd']}"

    agent = Agent(
        model=Model(id="gpt-4o", provider="openai"),
        system_prompt=custom_prompt,
        cwd="/custom/project",
    )

    prompt = agent.system_prompt
    # Path gets resolved, so /custom/project becomes C:\custom\project on Windows
    assert "Custom prompt for" in prompt
    assert "custom" in captured_ctx["cwd"].lower() or "project" in captured_ctx["cwd"].lower()
    assert captured_ctx["model"] is not None
    assert "tools" in captured_ctx
    assert "thinking_level" in captured_ctx

    # Callable overrides auto-injection
    assert "<env>" not in prompt, "Callable should override auto-injection"
    assert "<tools>" not in prompt, "Callable should override auto-injection"

    print("Test 3 PASSED: callable prompt with context dict")


def test_empty_tools_section():
    """No <tools> section when agent has no tools."""
    agent = Agent(
        model=Model(id="gpt-4o", provider="openai"),
        system_prompt="Test.",
    )

    prompt = agent.system_prompt
    assert "<tools>" not in prompt, "No tools section when no tools"

    print("Test 4 PASSED: no tools section when empty")


print("=" * 50)
print("Dynamic System Prompt Tests")
print("=" * 50 + "\n")

test_static_prompt_gets_auto_injected()
test_prompt_with_tools()
test_callable_prompt()
test_empty_tools_section()

print()
print("ALL SYSTEM PROMPT TESTS PASSED")
