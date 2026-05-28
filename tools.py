"""
Built-in tools for pi_agent: read_file, write_file, bash.

Usage:
    from pi_agent.tools import create_default_tools
    agent = Agent(model=..., tools=create_default_tools(cwd="."))
"""

from __future__ import annotations

import asyncio
import os
import shlex
import time
from pathlib import Path
from typing import Any

from .agent_types import AgentTool, AgentToolResult, TextContent, ImageContent


# ============================================================================
# read_file
# ============================================================================

def _create_read_file_tool(cwd: str) -> AgentTool:
    cwd_path = Path(cwd).resolve()

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        file_path = params.get("file_path", params.get("path", ""))
        if not file_path:
            return AgentToolResult(
                content=[TextContent(text="Error: file_path is required")],
                details={},
            )

        p = Path(file_path)
        if not p.is_absolute():
            p = cwd_path / p

        try:
            resolved = p.resolve()
            if not str(resolved).startswith(str(cwd_path)):
                return AgentToolResult(
                    content=[TextContent(text=f"Error: access denied - path outside working directory: {file_path}")],
                    details={},
                )
        except Exception:
            pass

        try:
            text = p.read_text(encoding="utf-8")
        except FileNotFoundError:
            return AgentToolResult(
                content=[TextContent(text=f"Error: file not found: {file_path}")],
                details={},
            )
        except PermissionError:
            return AgentToolResult(
                content=[TextContent(text=f"Error: permission denied: {file_path}")],
                details={},
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error reading {file_path}: {e}")],
                details={},
            )

        truncated = False
        max_chars = 50000
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[... {len(text) - max_chars} more characters truncated]"
            truncated = True

        return AgentToolResult(
            content=[TextContent(text=text)],
            details={"path": str(p), "size": len(text), "truncated": truncated},
        )

    return AgentTool(
        name="read",
        description="Read a file from the filesystem. Returns the file contents as text.",
        label="Read file",
        parametersSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read (absolute or relative to working directory)",
                },
            },
            "required": ["file_path"],
        },
        execute=execute,
    )


# ============================================================================
# write_file
# ============================================================================

def _create_write_file_tool(cwd: str) -> AgentTool:
    cwd_path = Path(cwd).resolve()

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        file_path = params.get("file_path", params.get("path", ""))
        content = params.get("content", "")

        if not file_path:
            return AgentToolResult(
                content=[TextContent(text="Error: file_path is required")],
                details={},
            )

        p = Path(file_path)
        if not p.is_absolute():
            p = cwd_path / p

        try:
            resolved = p.resolve()
            if not str(resolved).startswith(str(cwd_path)):
                return AgentToolResult(
                    content=[TextContent(text=f"Error: access denied - path outside working directory: {file_path}")],
                    details={},
                )
        except Exception:
            pass

        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except PermissionError:
            return AgentToolResult(
                content=[TextContent(text=f"Error: permission denied: {file_path}")],
                details={},
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error writing {file_path}: {e}")],
                details={},
            )

        return AgentToolResult(
            content=[TextContent(text=f"File written: {file_path} ({len(content)} characters)")],
            details={"path": str(p), "size": len(content)},
        )

    return AgentTool(
        name="write",
        description="Write content to a file. Creates parent directories if needed.",
        label="Write file",
        parametersSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write (absolute or relative to working directory)",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        },
        execute=execute,
    )


# ============================================================================
# bash
# ============================================================================

def _create_bash_tool(cwd: str) -> AgentTool:
    cwd_path = str(Path(cwd).resolve())

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        command = params.get("command", "")
        if not command:
            return AgentToolResult(
                content=[TextContent(text="Error: command is required")],
                details={},
            )

        timeout_val = params.get("timeout", 120)
        if not isinstance(timeout_val, (int, float)) or timeout_val <= 0:
            timeout_val = 120
        timeout_val = min(timeout_val, 600)  # cap at 10 minutes

        workdir = params.get("workdir", cwd_path)
        workdir = str(Path(workdir).resolve()) if workdir else cwd_path
        if not workdir.startswith(cwd_path):
            workdir = cwd_path

        env = os.environ.copy()

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env=env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_val,
                )
            except asyncio.TimeoutError:
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass
                return AgentToolResult(
                    content=[TextContent(text=f"Command timed out after {timeout_val}s:\n{command}")],
                    details={"command": command, "exitCode": -1, "timedOut": True},
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            output_parts: list[str] = []
            if stdout:
                output_parts.append(stdout)
            if stderr:
                output_parts.append(f"[stderr]\n{stderr}")

            output = "\n".join(output_parts) if output_parts else "(no output)"

            max_chars = 10000
            truncated = len(output) > max_chars
            if truncated:
                output = output[:max_chars] + f"\n\n[... {len(output) - max_chars} more characters truncated]"

            exit_code = process.returncode or 0
            status = f"exit code: {exit_code}" if exit_code != 0 else "ok"

            return AgentToolResult(
                content=[TextContent(text=f"Command: {command}\nStatus: {status}\n\n{output}")],
                details={
                    "command": command,
                    "exitCode": exit_code,
                    "truncated": truncated,
                    "timedOut": False,
                },
            )

        except FileNotFoundError:
            return AgentToolResult(
                content=[TextContent(text=f"Error: command not found: {shlex.split(command)[0] if command else ''}")],
                details={"command": command, "exitCode": -1},
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error executing command: {e}")],
                details={"command": command, "exitCode": -1},
            )

    return AgentTool(
        name="bash",
        description="Execute a shell command and return the output. Commands run in the working directory.",
        label="Run command",
        parametersSchema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (default: 120, max: 600)",
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory for the command (default: agent working directory)",
                },
            },
            "required": ["command"],
        },
        execute=execute,
    )


# ============================================================================
# Convenience: create all default tools
# ============================================================================

def create_default_tools(cwd: str = ".") -> list[AgentTool]:
    """Create the default set of tools: read, write, bash."""
    return [
        _create_read_file_tool(cwd),
        _create_write_file_tool(cwd),
        _create_bash_tool(cwd),
    ]
