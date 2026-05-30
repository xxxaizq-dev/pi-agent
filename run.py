"""
pi_agent CLI entry point.

Usage:
    set OPENAI_API_KEY=sk-...
    python -m pi_agent.run
    python pi_agent/run.py

    # or pass directly:
    python pi_agent/run.py --api-key sk-... --model gpt-4o

    # with working directory:
    python pi_agent/run.py --cwd E:/my-project
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

# Allow running from outside the project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pi_agent import Agent, Model, OpenAiProvider, Session, create_default_tools
from pi_agent.agent_types import make_user_message


def load_api_key(provider: str, cli_key: str | None) -> str:
    """Resolve API key: CLI arg > env var."""
    if cli_key:
        return cli_key
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    env_var = env_map.get(provider, "OPENAI_API_KEY")
    key = os.environ.get(env_var, "")
    if not key:
        print(f"[!] No API key found. Set {env_var} or pass --api-key")
        print(f"    Example: set {env_var}=sk-...")
        sys.exit(1)
    return key


async def main_loop(agent: Agent, cwd: str):
    """Simple interactive loop."""
    print(f"\npi_agent - cwd: {cwd}")
    print('Type "exit" or "quit" to stop, "compact" to trigger compaction.\n')

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if user_input.lower() == "compact":
            result = await agent.auto_compact()
            if result:
                print(f"[Compacted: {result.tokens_before} tokens before, "
                      f"{len(agent.messages)} messages now in context]")
            else:
                print("[Compaction not needed]")
            continue

        print()  # blank line before response
        await agent.prompt(user_input)
        print()  # blank line after response


async def main():
    parser = argparse.ArgumentParser(description="pi_agent - personal AI coding agent")
    parser.add_argument("--api-key", help="API key for the LLM provider")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"), help="Model ID")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"],
                        help="LLM provider (default: openai)")
    parser.add_argument("--cwd", default=".", help="Working directory (default: current)")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"), help="Override API base URL")
    parser.add_argument("--session", help="Session file path for persistence")
    parser.add_argument("--no-compaction", action="store_true", help="Disable auto-compaction")
    args = parser.parse_args()

    api_key = load_api_key(args.provider, args.api_key)
    cwd = str(Path(args.cwd).resolve())
    if not os.path.isdir(cwd):
        print(f"[!] Working directory not found: {cwd}")
        sys.exit(1)

    model_kwargs = {
        "id": args.model,
        "provider": args.provider,
        "contextWindow": 128000,
        "maxTokens": 4096,
    }
    if args.base_url:
        model_kwargs["baseUrl"] = args.base_url
    model = Model(**model_kwargs)

    provider = OpenAiProvider()

    # Session for persistence
    session = None
    if args.session:
        session_path = Path(args.session)
        if session_path.exists():
            session = await Session.open(session_path)
            print(f"[Session loaded: {session_path}]")
        else:
            session = await Session.create(session_path, cwd=cwd)
            print(f"[Session created: {session_path}]")
    else:
        # Default session in temp dir
        import tempfile
        tmp_dir = Path(tempfile.gettempdir()) / "pi_agent_sessions"
        tmp_dir.mkdir(exist_ok=True)
        session_path = tmp_dir / f"{args.model.replace('/', '_')}.jsonl"
        if session_path.exists():
            session = await Session.open(session_path)
        else:
            session = await Session.create(session_path, cwd=cwd)

    # Build agent
    agent_kwargs: dict = dict(
        model=model,
        provider=provider,
        tools=create_default_tools(cwd=cwd),
        cwd=cwd,
        system_prompt=(
            "You are a helpful AI coding assistant. "
            "You can read files, write files, and execute shell commands. "
            "When given a task, think step by step and use your tools to accomplish it. "
            "Before writing code, understand the existing project structure by reading files first."
        ),
        api_key=api_key,
        session=session,
    )
    if args.no_compaction:
        from pi_agent.agent_types import CompactionSettings
        agent_kwargs["compaction_settings"] = CompactionSettings(enabled=False)

    agent = Agent(**agent_kwargs)

    # Subscribe to events for streaming output
    streamed_text = False
    def on_event(event: dict):
        nonlocal streamed_text
        etype = event.get("type", "")
        if etype == "message_update":
            text = event.get("assistantMessageEvent", {}).get("text", "")
            if text:
                streamed_text = True
                sys.stdout.write(text)
                sys.stdout.flush()
        elif etype == "message_end":
            msg = event.get("message")
            if msg and getattr(msg, "role", None) == "assistant":
                if not streamed_text:
                    for block in getattr(msg, "content", []) or []:
                        if getattr(block, "type", None) == "text":
                            sys.stdout.write(getattr(block, "text", ""))
                    sys.stdout.flush()
                streamed_text = False
                sys.stdout.write("\n")
                sys.stdout.flush()
        elif etype == "turn_start":
            streamed_text = False
        elif etype == "tool_execution_start":
            name = event.get("toolName", "?")
            sys.stdout.write(f"\n[Running: {name}...]")
            sys.stdout.flush()
        elif etype == "tool_execution_end":
            name = event.get("toolName", "?")
            is_err = event.get("isError", False)
            status = "error" if is_err else "done"
            sys.stdout.write(f"[{name}: {status}]\n")
            sys.stdout.flush()

    agent.on_event(on_event)

    await main_loop(agent, cwd)


if __name__ == "__main__":
    asyncio.run(main())
