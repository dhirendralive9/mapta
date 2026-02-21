"""
MAPTA - Multi-Agent Penetration Testing Assistant
Patched for LiteLLM (Chat Completions API) — model-agnostic support.

Changes from original:
  - Replaced AsyncOpenAI Responses API → litellm.acompletion() (Chat Completions)
  - Model configurable via MAPTA_MODEL env var (default: gpt-4o)
  - Sandbox model configurable via MAPTA_SANDBOX_MODEL env var (falls back to MAPTA_MODEL)
  - Extended thinking support via MAPTA_REASONING_EFFORT env var
  - Anthropic thinking budget configurable via MAPTA_THINKING_BUDGET_TOKENS
  - Tool schema converted to Chat Completions format
  - Message format converted (developer→system, input_text→text)
  - Response parsing converted (response.output→response.choices[0].message)
  - Tool result format converted (function_call_output→tool role)
  - All 3 agent loops patched: main, sandbox, validator

Supports: OpenAI, Anthropic, AWS Bedrock, Azure, Ollama, vLLM, etc.
Set provider via env vars:
  MAPTA_MODEL=anthropic/claude-sonnet-4-20250514
  MAPTA_MODEL=bedrock/anthropic.claude-sonnet-4-20250514-v1:0
  MAPTA_MODEL=openai/gpt-4o
  MAPTA_MODEL=ollama/llama3

Extended thinking:
  MAPTA_REASONING_EFFORT=high          # low|medium|high|none (default: none)
  MAPTA_THINKING_BUDGET_TOKENS=10000   # Anthropic thinking budget (default: 8192)
  
  LiteLLM auto-translates reasoning_effort per provider:
    - OpenAI (o3/gpt-5)  → reasoning={"effort": "high"}
    - Anthropic (Claude)  → thinking={"type":"enabled","budget_tokens":N}
    - Gemini              → thinking_budget / thinking_level
    - DeepSeek            → native reasoning
    - Other models        → param silently dropped (drop_params=True)
"""

import os
import json
import asyncio
from typing import Any, Dict, Optional, List
from datetime import datetime, UTC
import threading
import logging
import importlib

from function_tool import function_tool
import json as json_module
import httpx
import aiohttp

# LiteLLM import — replaces AsyncOpenAI
import litellm

# Optional: reduce litellm verbosity
litellm.set_verbose = False

# Auto-handle Anthropic thinking+tool_calling compatibility
# When thinking is enabled but assistant messages lack thinking_blocks
# (e.g. from previous turns), LiteLLM will auto-drop thinking param
# for that turn instead of crashing. Essential for multi-turn tool use.
litellm.modify_params = True

# --- Model Configuration ---
MAPTA_MODEL = os.getenv("MAPTA_MODEL", "gpt-4o")
MAPTA_SANDBOX_MODEL = os.getenv("MAPTA_SANDBOX_MODEL", MAPTA_MODEL)

# --- Extended Thinking Configuration ---
# "none" = disabled (default), "low"/"medium"/"high" = enabled
MAPTA_REASONING_EFFORT = os.getenv("MAPTA_REASONING_EFFORT", "none").lower()
# Anthropic-specific: explicit thinking budget in tokens (min 1024)
MAPTA_THINKING_BUDGET_TOKENS = int(os.getenv("MAPTA_THINKING_BUDGET_TOKENS", "8192"))


def _build_reasoning_kwargs(model: str) -> dict:
    """Build provider-appropriate extended thinking kwargs for litellm.acompletion().
    
    Returns a dict of extra kwargs to spread into the acompletion() call.
    Returns empty dict if thinking is disabled.
    
    Strategy:
      1. reasoning_effort is the universal LiteLLM parameter — works cross-provider
      2. For Anthropic, we ALSO pass the native thinking param for precise budget control
      3. drop_params=True ensures unsupported providers silently ignore these
    """
    if MAPTA_REASONING_EFFORT == "none":
        return {}
    
    kwargs = {
        "reasoning_effort": MAPTA_REASONING_EFFORT,  # Universal — LiteLLM translates per provider
        "drop_params": True,  # Silently ignore on providers that don't support it
    }
    
    # For Anthropic models: also pass native thinking param for precise budget control
    # reasoning_effort alone works, but thinking={} gives us token budget granularity
    model_lower = model.lower()
    is_anthropic = any(x in model_lower for x in ["anthropic/", "claude", "bedrock/anthropic"])
    
    if is_anthropic and MAPTA_REASONING_EFFORT in ("low", "medium", "high"):
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": MAPTA_THINKING_BUDGET_TOKENS,
        }
    
    return kwargs


SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "#security-alerts")


# Global sandbox configuration (sanitized for open release)
# Provide a factory via env var SANDBOX_FACTORY="your_module:create_sandbox" that returns a sandbox instance
SANDBOX_FACTORY = os.getenv("SANDBOX_FACTORY")

# Thread-local storage for sandbox instances
_thread_local = threading.local()

def get_current_sandbox():
    """Get the sandbox instance for the current thread/scan."""
    return getattr(_thread_local, 'sandbox', None)

def set_current_sandbox(sandbox):
    """Set the sandbox instance for the current thread/scan."""
    _thread_local.sandbox = sandbox

def create_sandbox_from_env():
    """Create a sandbox instance using a user-provided factory specified in SANDBOX_FACTORY.

    SANDBOX_FACTORY should be in the form "module_path:function_name" and must return an
    object exposing .files.write(path, content), .commands.run(cmd, timeout=..., user=...),
    and optional .set_timeout(ms) and .kill().

    Returns None if not configured.
    """
    factory_path = SANDBOX_FACTORY
    if not factory_path:
        logging.info("Sandbox factory not configured; running without a sandbox.")
        return None
    try:
        module_name, func_name = factory_path.rsplit(":", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, func_name)
        sandbox = factory()
        # Optionally extend timeout if provider supports it
        if hasattr(sandbox, "set_timeout"):
            try:
                sandbox.set_timeout(timeout=12000)
            except TypeError:
                sandbox.set_timeout(12000)
        return sandbox
    except Exception as exc:
        logging.warning(f"Failed to create sandbox from SANDBOX_FACTORY: {exc}")
        return None

# Usage tracking
class UsageTracker:
    def __init__(self):
        self.main_agent_usage = []
        self.sandbox_agent_usage = []
        self.start_time = datetime.now(UTC)
    
    def log_main_agent_usage(self, usage_data, target_url=""):
        """Log usage data from main agent responses."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "target_url": target_url,
            "agent_type": "main_agent",
            "usage": usage_data
        }
        self.main_agent_usage.append(entry)
        logging.info(f"Main Agent Usage - Target: {target_url}, Usage: {usage_data}")
    
    def log_sandbox_agent_usage(self, usage_data, target_url=""):
        """Log usage data from sandbox agent responses."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "target_url": target_url,
            "agent_type": "sandbox_agent", 
            "usage": usage_data
        }
        self.sandbox_agent_usage.append(entry)
        logging.info(f"Sandbox Agent Usage - Target: {target_url}, Usage: {usage_data}")
    
    def get_summary(self):
        """Get usage summary for all agents."""
        return {
            "scan_duration": str(datetime.now(UTC) - self.start_time),
            "main_agent_calls": len(self.main_agent_usage),
            "sandbox_agent_calls": len(self.sandbox_agent_usage),
            "total_calls": len(self.main_agent_usage) + len(self.sandbox_agent_usage),
            "main_agent_usage": self.main_agent_usage,
            "sandbox_agent_usage": self.sandbox_agent_usage
        }
    
    def save_to_file(self, filename_prefix=""):
        """Save usage data to JSON file."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}usage_log_{timestamp}.json"
        
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(self.get_summary(), f, indent=2, default=str)
        
        logging.info(f"Usage data saved to {filename}")
        return filename

# Thread-local storage for usage trackers
def get_current_usage_tracker():
    """Get the usage tracker for the current thread/scan."""
    return getattr(_thread_local, 'usage_tracker', None)

def set_current_usage_tracker(tracker):
    """Set the usage tracker for the current thread/scan."""
    _thread_local.usage_tracker = tracker


# ============================================================
# Chat Completions helper — converts tool_calls from LiteLLM
# response into a unified format for execute_function_call()
# ============================================================

class ToolCallShim:
    """Shim to give Chat Completions tool_calls the same interface
    that execute_function_call() expects (.name, .arguments, .call_id)."""
    def __init__(self, tool_call):
        self.name = tool_call.function.name
        self.arguments = tool_call.function.arguments
        self.call_id = tool_call.id


async def execute_function_call(function_call):
    """Execute a tool call and return a Chat Completions tool-result message."""
    function_call_arguments = json.loads(function_call.arguments)

    # Execute the function logic
    result = await execute_tool(function_call.name, function_call_arguments)

    # Return in Chat Completions tool-result format
    return {
        "role": "tool",
        "tool_call_id": function_call.call_id,
        "content": result if isinstance(result, str) else json.dumps(result),
    }


# In-memory store: email -> JWT token (for mail.tm API)
email_token_store = {}



@function_tool
async def get_registered_emails():
    """
    Return the list of email accounts in case you need to use them to receive emails such as account activation emails, credentials, etc.
    """
    return json_module.dumps(list(email_token_store.keys()))



@function_tool
async def list_account_messages(email: str, limit: int = 50):
    """
    List recent messages for the given email account.
    Returns JSON list: [{id, subject, from, intro, seen, createdAt}]
    
    Args:
        email: The email account to fetch messages for
        limit: Maximum number of messages to return (default: 50)
    """
    jwt = email_token_store.get(email)
    if not jwt:
        return f"No JWT token stored for {email}. Call set_email_jwt_token(email, jwt_token) first."

    headers = {"Authorization": f"Bearer {jwt}"}
    try:
        with httpx.Client(timeout=30) as http_client:
            resp = http_client.get("https://api.mail.tm/messages", headers=headers)
            if resp.status_code != 200:
                return f"Failed to fetch messages. Status: {resp.status_code}, Response: {resp.text}"
            data = resp.json()
            messages = data.get("hydra:member", [])
            items = []
            for m in messages[:limit]:
                sender = m.get("from") or {}
                items.append(
                    {
                        "id": m.get("id"),
                        "subject": m.get("subject"),
                        "from": sender.get("address") or sender.get("name") or "",
                        "intro": m.get("intro", ""),
                        "seen": m.get("seen", False),
                        "createdAt": m.get("createdAt", ""),
                    }
                )
            return json_module.dumps(items)
    except Exception as e:
        return f"Request failed: {e}"



@function_tool
async def get_message_by_id(email: str, message_id: str):
    """
    Fetch a specific message by id for the given email account using its stored JWT.
    Returns JSON: {id, subject, from, text, html}
    
    Args:
        email: The email account to fetch the message from
        message_id: The ID of the message to fetch
    """
    jwt = email_token_store.get(email)
    if not jwt:
        return f"No JWT token stored for {email}. Call set_email_jwt_token(email, jwt_token) first."

    headers = {"Authorization": f"Bearer {jwt}"}
    try:
        with httpx.Client(timeout=30) as http_client:
            resp = http_client.get(
                f"https://api.mail.tm/messages/{message_id}", headers=headers
            )
            if resp.status_code != 200:
                return f"Failed to fetch message. Status: {resp.status_code}, Response: {resp.text}"
            msg = resp.json()
            sender = msg.get("from") or {}
            result = {
                "id": msg.get("id"),
                "subject": msg.get("subject"),
                "from": sender.get("address") or sender.get("name") or "",
                "text": msg.get("text", ""),
                "html": msg.get("html", ""),
            }
            return json_module.dumps(result)
    except Exception as e:
        return f"Request failed: {e}"


@function_tool(name_override="send_slack_alert")
async def send_slack_security_alert(
    vulnerability_type: str,
    severity: str,
    target_url: str,
    description: str,
    evidence: Optional[str] = None,
    recommendation: Optional[str] = None,
    thread_ts: Optional[str] = None
):
    """
    Send a security vulnerability alert to Slack channel.
    
    Args:
        vulnerability_type: Type of vulnerability (e.g., "XSS", "SQL Injection", "IDOR")
        severity: Severity level ("Critical", "High", "Medium", "Low", "Info")
        target_url: The affected URL or endpoint
        description: Detailed description of the vulnerability
        evidence: Optional proof-of-concept or evidence details
        recommendation: Optional remediation recommendation
        thread_ts: Optional thread timestamp to reply to existing thread
    """
    
    severity_colors = {
        "Critical": "#FF0000",
        "High": "#FF6600",
        "Medium": "#FFB84D",
        "Low": "#FFCC00",
        "Info": "#0099FF"
    }
    
    severity_emojis = {
        "Critical": "🚨",
        "High": "⚠️",
        "Medium": "⚡",
        "Low": "📝",
        "Info": "ℹ️"
    }
    
    color = severity_colors.get(severity, "#808080")
    emoji = severity_emojis.get(severity, "📌")
    
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} {vulnerability_type} Vulnerability Detected",
                "emoji": True
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Severity:*\n{severity}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Target:*\n<{target_url}|{target_url}>"
                }
            ]
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Description:*\n{description}"
            }
        }
    ]
    
    if evidence:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Evidence/PoC:*\n```{evidence[:500]}```"
            }
        })
    
    if recommendation:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Recommendation:*\n{recommendation}"
            }
        })
    
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"Detected at: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
            }
        ]
    })
    
    fallback_text = f"{emoji} {severity} {vulnerability_type} vulnerability found at {target_url}"
    
    if SLACK_WEBHOOK_URL:
        payload = {
            "channel": SLACK_CHANNEL,
            "username": "Security Scanner Bot",
            "icon_emoji": ":shield:",
            "text": fallback_text,
            "blocks": blocks,
            "attachments": [
                {
                    "color": color,
                    "fallback": fallback_text
                }
            ]
        }
        
        if thread_ts:
            payload["thread_ts"] = thread_ts
        
        async with aiohttp.ClientSession() as session:
            async with session.post(SLACK_WEBHOOK_URL, json=payload) as response:
                if response.status == 200:
                    return json_module.dumps({"success": True, "message": "Alert sent to Slack successfully"})
                else:
                    error_text = await response.text()
                    return json_module.dumps({"success": False, "error": f"Failed to send to Slack: {error_text}"})
    else:
        return json_module.dumps({
            "success": False, 
            "error": "No Slack webhook configured. Set SLACK_WEBHOOK_URL in .env file"
        })


@function_tool(name_override="send_slack_summary")
async def send_slack_scan_summary(
    target_url: str,
    total_findings: int,
    critical_count: int = 0,
    high_count: int = 0,
    medium_count: int = 0,
    low_count: int = 0,
    scan_duration: Optional[str] = None
):
    """
    Send a summary of the security scan to Slack.
    
    Args:
        target_url: The target that was scanned
        total_findings: Total number of vulnerabilities found
        critical_count: Number of critical severity findings
        high_count: Number of high severity findings
        medium_count: Number of medium severity findings
        low_count: Number of low severity findings
        scan_duration: Optional duration of the scan
    """
    
    if critical_count > 0:
        status_emoji = "🔴"
        status_text = "Critical Issues Found"
        color = "#FF0000"
    elif high_count > 0:
        status_emoji = "🟠"
        status_text = "High Risk Issues Found"
        color = "#FF6600"
    elif medium_count > 0:
        status_emoji = "🟡"
        status_text = "Medium Risk Issues Found"
        color = "#FFB84D"
    elif low_count > 0:
        status_emoji = "🟢"
        status_text = "Low Risk Issues Found"
        color = "#00FF00"
    else:
        status_emoji = "✅"
        status_text = "No Issues Found"
        color = "#00FF00"
    
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{status_emoji} Security Scan Summary",
                "emoji": True
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Target:* <{target_url}|{target_url}>\n*Status:* {status_text}\n*Total Findings:* {total_findings}"
            }
        }
    ]
    
    if total_findings > 0:
        findings_text = []
        if critical_count > 0:
            findings_text.append(f"🚨 Critical: {critical_count}")
        if high_count > 0:
            findings_text.append(f"⚠️ High: {high_count}")
        if medium_count > 0:
            findings_text.append(f"⚡ Medium: {medium_count}")
        if low_count > 0:
            findings_text.append(f"📝 Low: {low_count}")
        
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Findings Breakdown:*\n" + "\n".join(findings_text)
            }
        })
    
    if scan_duration:
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Scan Duration: {scan_duration} | Completed: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
                }
            ]
        })
    
    fallback_text = f"{status_emoji} Security scan completed for {target_url}: {total_findings} findings"
    
    if SLACK_WEBHOOK_URL:
        payload = {
            "channel": SLACK_CHANNEL,
            "username": "Security Scanner Bot",
            "icon_emoji": ":shield:",
            "text": fallback_text,
            "blocks": blocks,
            "attachments": [
                {
                    "color": color,
                    "fallback": fallback_text
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(SLACK_WEBHOOK_URL, json=payload) as response:
                if response.status == 200:
                    return json_module.dumps({"success": True, "message": "Summary sent to Slack successfully"})
                else:
                    error_text = await response.text()
                    return json_module.dumps({"success": False, "error": f"Failed to send to Slack: {error_text}"})
    else:
        return json_module.dumps({
            "success": False,
            "error": "No Slack webhook configured. Set SLACK_WEBHOOK_URL in .env file"
        })


# ============================================================
# Agent loops — rewritten for Chat Completions API via LiteLLM
# ============================================================

def _extract_text_from_response(response) -> str:
    """Extract text content from a Chat Completions response."""
    msg = response.choices[0].message
    return msg.content or ""


def _extract_tool_calls(response) -> list:
    """Extract tool calls from a Chat Completions response, wrapped in ToolCallShim."""
    msg = response.choices[0].message
    if not msg.tool_calls:
        return []
    return [ToolCallShim(tc) for tc in msg.tool_calls]


def _build_assistant_message(response) -> dict:
    """Build the assistant message dict to append to conversation history.
    Includes tool_calls if present, so the model sees its own requests.
    
    CRITICAL for Anthropic thinking+tools: preserves thinking_blocks in
    provider_specific_fields so Anthropic sees them on subsequent turns.
    Without this, Anthropic rejects the request with:
    "Expected `thinking` or `redacted_thinking`, but found `text`"
    """
    msg = response.choices[0].message
    assistant_msg = {"role": "assistant", "content": msg.content or ""}
    
    # Preserve thinking_blocks for Anthropic multi-turn (if present)
    # LiteLLM returns these in message.thinking_blocks or provider_specific_fields
    if hasattr(msg, 'thinking_blocks') and msg.thinking_blocks:
        assistant_msg["thinking_blocks"] = msg.thinking_blocks
    
    if msg.tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            }
            for tc in msg.tool_calls
        ]
        # Some providers require content=None when tool_calls present
        if not msg.content:
            assistant_msg["content"] = None
    return assistant_msg


@function_tool(name_override="sandbox_agent")
async def run_sandbox_agent(instruction: str, max_rounds: int = 100):
    """
    Nested agent loop that uses only sandbox execution tools to fulfill the provided instruction.
    Returns the final textual response when the model stops requesting tools or when max_rounds is hit.
    
    Args:
        instruction: The instruction for the sandbox agent to execute
        max_rounds: Maximum number of execution rounds (default: 100)
    """
    sandbox_system_prompt = os.getenv(
        "SANDBOX_SYSTEM_PROMPT",
        (
            "You are an agent that autonomously interacts with an isolated sandbox using two tools: "
            "`sandbox_run_command` (bash) and `sandbox_run_python` (Python). Keep responses within 30,000 "
            "characters; chunk large outputs. Think step-by-step before taking actions."
        ),
    )

    messages = [
        {"role": "system", "content": sandbox_system_prompt},
        {"role": "user", "content": instruction},
    ]

    # Restrict to low-level sandbox tools
    sandbox_tools = [
        t for t in tools_chat_completions if t["function"]["name"] in ("sandbox_run_command", "sandbox_run_python")
    ]

    rounds_completed = 0
    while True:
        response = await litellm.acompletion(
            model=MAPTA_SANDBOX_MODEL,
            messages=messages,
            tools=sandbox_tools,
            tool_choice="auto",
            **_build_reasoning_kwargs(MAPTA_SANDBOX_MODEL),
        )

        # Log sandbox agent usage
        usage_tracker = get_current_usage_tracker()
        if usage_tracker and hasattr(response, 'usage') and response.usage:
            usage_tracker.log_sandbox_agent_usage(
                response.usage.model_dump() if hasattr(response.usage, 'model_dump') else str(response.usage),
                getattr(_thread_local, 'current_target_url', '')
            )

        tool_calls = _extract_tool_calls(response)

        if not tool_calls:
            return _extract_text_from_response(response)

        # Append assistant message with tool_calls to history
        messages.append(_build_assistant_message(response))

        # Execute tool calls in parallel
        shims = tool_calls
        tasks = [execute_function_call(shim) for shim in shims]
        results = await asyncio.gather(*tasks)

        # Append tool results to history
        messages.extend(results)
        rounds_completed += 1

        if max_rounds and rounds_completed >= max_rounds:
            return f"[sandbox_agent] Reached max rounds limit: {max_rounds}"

        
@function_tool(name_override="validator_agent")
async def run_validator_agent(instruction: str, max_rounds: int = 50):
    """
    Agent loop specialized for validating Proofs-of-Concept (PoCs) in the sandbox.
    Use only sandbox tools, keep outputs concise, and return a clear verdict.

    Args:
        instruction: Validation instruction that includes the PoC and expected outcome
        max_rounds: Maximum number of execution rounds (default: 50)
    """
    validator_system_prompt = os.getenv(
        "VALIDATOR_SYSTEM_PROMPT",
        (
            "You validate security PoCs inside an isolated sandbox using two tools: "
            "`sandbox_run_command` (bash) and `sandbox_run_python` (Python). Your goal is to: "
            "(1) Reproduce the PoC minimally and safely, (2) Capture evidence (stdout, file diffs, HTTP responses), "
            "(3) Decide if the PoC reliably demonstrates a real vulnerability with impact, (4) Provide a concise verdict. "
            "Always think step-by-step before actions. Keep outputs within 30,000 characters and chunk large outputs. "
            "Avoid destructive actions unless explicitly required for validation."
        ),
    )

    messages = [
        {"role": "system", "content": validator_system_prompt},
        {"role": "user", "content": instruction},
    ]

    validator_tools = [
        t for t in tools_chat_completions if t["function"]["name"] in ("sandbox_run_command", "sandbox_run_python")
    ]

    rounds_completed = 0
    while True:
        response = await litellm.acompletion(
            model=MAPTA_SANDBOX_MODEL,
            messages=messages,
            tools=validator_tools,
            tool_choice="auto",
            **_build_reasoning_kwargs(MAPTA_SANDBOX_MODEL),
        )

        # Reuse sandbox usage tracker for validator agent
        usage_tracker = get_current_usage_tracker()
        if usage_tracker and hasattr(response, 'usage') and response.usage:
            usage_tracker.log_sandbox_agent_usage(
                response.usage.model_dump() if hasattr(response.usage, 'model_dump') else str(response.usage),
                getattr(_thread_local, 'current_target_url', '')
            )

        tool_calls = _extract_tool_calls(response)

        if not tool_calls:
            return _extract_text_from_response(response)

        messages.append(_build_assistant_message(response))
        tasks = [execute_function_call(shim) for shim in tool_calls]
        results = await asyncio.gather(*tasks)
        messages.extend(results)
        rounds_completed += 1

        if max_rounds and rounds_completed >= max_rounds:
            return f"[validator_agent] Reached max rounds limit: {max_rounds}"

        
@function_tool
async def sandbox_run_python(python_code: str, timeout: int = 120):
    """
    Run Python code inside a Docker sandbox and return stdout/stderr/exit code. If the output exceeds 30000 characters, output will be truncated before being returned to you.

    Args:
        python_code: Python code to execute (e.g., "print('Hello World')").
        timeout: Max seconds to wait before timing out the code execution.

    Returns:
        A string containing exit code, stdout, and stderr.
    """

    print(f"Running Python code: {python_code[:100]}...")
    try:
        sbx = get_current_sandbox()
        if sbx is None:
            return "Error: No sandbox instance available for this scan"
            
        import uuid
        script_name = f"temp_script_{uuid.uuid4().hex[:8]}.py"
        script_path = f"/home/user/{script_name}"
        
        sbx.files.write(script_path, python_code)
        result = sbx.commands.run(f"source .venv/bin/activate && python3 {script_path}", timeout=timeout, user="root")

        stdout_raw = (
            result.stdout
            if hasattr(result, "stdout") and result.stdout is not None
            else ""
        )
        stderr_raw = (
            result.stderr
            if hasattr(result, "stderr") and result.stderr is not None
            else ""
        )
        exit_code = result.exit_code if hasattr(result, "exit_code") else "unknown"

        output = f"Exit code: {exit_code}\n\nSTDOUT\n{stdout_raw}\n\nSTDERR\n{stderr_raw}"

        if len(output) > 30000:
            output = (
                output[:30000]
                + "\n...[OUTPUT TRUNCATED - EXCEEDED 30000 CHARACTERS]"
            )

        return output
    except Exception as e:
        return f"Failed to run Python code in sandbox: {e}"


@function_tool
async def sandbox_run_command(command: str, timeout: int = 120):
    """
    Run a shell command inside an ephemeral sandbox and return stdout/stderr/exit code.

    Arguments:
        command: Shell command to execute (e.g., "ls -la").
        timeout: Max seconds to wait before timing out the command.

    Returns:
        A string containing exit code, stdout, and stderr.
    """

    print(f"Running command: {command}")
    try:
        sbx = get_current_sandbox()
        if sbx is None:
            return "Error: No sandbox instance available for this scan"
            
        result = sbx.commands.run(command, timeout=timeout, user="root")

        stdout_raw = (
            result.stdout
            if hasattr(result, "stdout") and result.stdout is not None
            else ""
        )
        stderr_raw = (
            result.stderr
            if hasattr(result, "stderr") and result.stderr is not None
            else ""
        )
        exit_code = result.exit_code if hasattr(result, "exit_code") else "unknown"

        return f"Exit code: {exit_code}\n\nSTDOUT\n{stdout_raw}\n\nSTDERR\n{stderr_raw}"
    except Exception as e:
        return f"Failed to run command in sandbox: {e}"

# Collect all function tools that were decorated
_function_tools = {
    "sandbox_run_command": sandbox_run_command,
    "sandbox_run_python": sandbox_run_python,
    "sandbox_agent": run_sandbox_agent,
    "validator_agent": run_validator_agent,
    "get_message_by_id": get_message_by_id,
    "list_account_messages": list_account_messages,
    "get_registered_emails": get_registered_emails,
    "send_slack_alert": send_slack_security_alert,
    "send_slack_summary": send_slack_scan_summary,
}

async def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    try:
        if name in _function_tools:
            func_tool = _function_tools[name]
            if name == "sandbox_agent":
                instruction = arguments.get("instruction", arguments.get("input", ""))
                max_rounds = arguments.get("max_rounds", 100)
                out = await func_tool(instruction, max_rounds)
            else:
                out = await func_tool(**arguments)
        else:
            out = {"error": f"Unknown tool: {name}", "args": arguments}
    except Exception as e:
        out = {"error": str(e), "args": arguments}
    return json.dumps(out) if not isinstance(out, str) else out


# ============================================================
# Tool schema generation — Chat Completions format
# ============================================================

def generate_tools_responses_api():
    """Generate tools in OpenAI Responses API format (for reference/compatibility)."""
    tools = []
    for _, func_tool in _function_tools.items():
        if hasattr(func_tool, 'name') and hasattr(func_tool, 'description') and hasattr(func_tool, 'params_json_schema'):
            tool_def = {
                "type": "function",
                "name": func_tool.name,
                "description": func_tool.description,
                "parameters": func_tool.params_json_schema,
                "strict": getattr(func_tool, 'strict_json_schema', True),
            }
            tools.append(tool_def)
    return tools


def generate_tools_chat_completions():
    """Generate tools in Chat Completions format for LiteLLM.
    
    Chat Completions format:
    {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    
    vs Responses API format:
    {"type": "function", "name": ..., "description": ..., "parameters": ...}
    """
    tools = []
    for _, func_tool in _function_tools.items():
        if hasattr(func_tool, 'name') and hasattr(func_tool, 'description') and hasattr(func_tool, 'params_json_schema'):
            tool_def = {
                "type": "function",
                "function": {
                    "name": func_tool.name,
                    "description": func_tool.description,
                    "parameters": func_tool.params_json_schema,
                }
            }
            tools.append(tool_def)
    return tools


# Generate both formats
tools_responses_api = generate_tools_responses_api()        # Original format (kept for reference)
tools_chat_completions = generate_tools_chat_completions()  # LiteLLM format (used at runtime)


def read_targets_from_file(file_path: str) -> List[str]:
    """
    Read target URLs from a text file, one per line.
    Ignores empty lines and lines starting with #.
    """
    targets = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    targets.append(line)
        return targets
    except FileNotFoundError:
        print(f"Error: Target file '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading target file: {e}")
        return []


async def run_continuously(max_rounds: int = 100, user_prompt: str = "", system_prompt: str = "", target_url: str = "", sandbox_instance=None):
    """
    Main agent loop — Chat Completions API via LiteLLM.
    
    Keep prompting the model and executing any requested tool calls in parallel
    until the model stops requesting tools or the optional max_rounds is reached.
    """
    # Create sandbox instance if not provided
    if sandbox_instance is None:
        sandbox_instance = create_sandbox_from_env()
    
    set_current_sandbox(sandbox_instance)
    _thread_local.current_target_url = target_url
    
    rounds_completed = 0

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Main agent tools — exclude low-level sandbox tools (those are for nested agents only)
    main_agent_tool_names = {"sandbox_agent", "validator_agent", "get_message_by_id", "list_account_messages", "get_registered_emails", "send_slack_alert", "send_slack_summary"}
    main_agent_tools = [
        t for t in tools_chat_completions if t["function"]["name"] in main_agent_tool_names
    ]

    try:
        while True:
            # 1) Ask the model what to do next
            response = await litellm.acompletion(
                model=MAPTA_MODEL,
                messages=messages,
                tools=main_agent_tools,
                tool_choice="auto",
                **_build_reasoning_kwargs(MAPTA_MODEL),
            )

            # Log main agent usage
            usage_tracker = get_current_usage_tracker()
            if usage_tracker and hasattr(response, 'usage') and response.usage:
                usage_tracker.log_main_agent_usage(
                    response.usage.model_dump() if hasattr(response.usage, 'model_dump') else str(response.usage),
                    target_url
                )

            # 2) Check for tool calls
            tool_calls = _extract_tool_calls(response)

            # If no tool calls, print result and stop
            if not tool_calls:
                output_text = _extract_text_from_response(response)
                print(output_text)
                return output_text

            # 3) Append assistant message and execute tool calls in parallel
            messages.append(_build_assistant_message(response))

            print(f"[debug] Executing {len(tool_calls)} function calls in parallel...")
            tasks = [execute_function_call(shim) for shim in tool_calls]
            results = await asyncio.gather(*tasks)

            # 4) Add tool results for next round
            messages.extend(results)
            rounds_completed += 1

            # 5) Safety valve
            if max_rounds and rounds_completed >= max_rounds:
                print(f"[debug] Reached max rounds limit: {max_rounds}")
                break
    finally:
        if sandbox_instance and hasattr(sandbox_instance, "kill"):
            sandbox_instance.kill()

async def run_single_target_scan(target_url: str, system_prompt: str, base_user_prompt: str, max_rounds: int = 100):
    """
    Run a security scan for a single target URL.
    Returns the scan result and saves it to a file.
    Each scan gets its own isolated sandbox instance.
    """
    print(f"Starting scan for: {target_url}")
    
    sandbox_instance = create_sandbox_from_env()
    usage_tracker = UsageTracker()
    set_current_usage_tracker(usage_tracker)
    
    user_prompt = base_user_prompt.format(target_url=target_url)
    
    try:
        result = await run_continuously(
            user_prompt=user_prompt, 
            system_prompt=system_prompt, 
            target_url=target_url,
            max_rounds=max_rounds,
            sandbox_instance=sandbox_instance
        )
        
        filename = target_url.replace("https://", "").replace("http://", "").replace("/", "_") + ".md"
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write(result)
        
        site_name = target_url.replace("https://", "").replace("http://", "").split('/')[0]
        usage_filename = usage_tracker.save_to_file(f"{site_name}_")
        
        print(f"Scan completed for {target_url} - Results saved to {filename}")
        print(f"Usage data saved to {usage_filename}")
        
        return {
            "target": target_url,
            "filename": filename,
            "usage_filename": usage_filename,
            "status": "completed",
            "result": result,
            "usage_summary": usage_tracker.get_summary()
        }
        
    except Exception as e:
        print(f"Error scanning {target_url}: {e}")
        return {
            "target": target_url,
            "filename": None,
            "status": "error",
            "error": str(e)
        }

async def run_parallel_scans(targets: List[str], system_prompt: str, base_user_prompt: str, max_rounds: int = 100):
    """
    Run security scans for multiple targets in parallel.
    """
    print(f"Starting parallel scans for {len(targets)} targets...")
    print(f"Using model: {MAPTA_MODEL} (sandbox: {MAPTA_SANDBOX_MODEL})")
    
    tasks = [
        run_single_target_scan(target, system_prompt, base_user_prompt, max_rounds)
        for target in targets
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    completed = 0
    errors = 0
    
    for result in results:
        if isinstance(result, Exception):
            print(f"Task failed with exception: {result}")
            errors += 1
        elif result.get("status") == "completed":
            completed += 1
        else:
            errors += 1
    
    print(f"\nScan Summary:")
    print(f"Total targets: {len(targets)}")
    print(f"Completed successfully: {completed}")
    print(f"Failed: {errors}")
    
    total_main_calls = 0
    total_sandbox_calls = 0
    usage_files = []
    
    for result in results:
        if isinstance(result, dict) and result.get("status") == "completed":
            summary = result.get("usage_summary", {})
            total_main_calls += summary.get("main_agent_calls", 0)
            total_sandbox_calls += summary.get("sandbox_agent_calls", 0)
            if "usage_filename" in result:
                usage_files.append(result["usage_filename"])
    
    print(f"\nUsage Summary:")
    print(f"Model: {MAPTA_MODEL}")
    print(f"Total Main Agent API calls: {total_main_calls}")
    print(f"Total Sandbox Agent API calls: {total_sandbox_calls}")
    print(f"Total API calls: {total_main_calls + total_sandbox_calls}")
    print(f"Usage files created: {len(usage_files)}")
    for uf in usage_files:
        print(f"  - {uf}")
    
    return results


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('scan_usage.log'),
            logging.StreamHandler()
        ]
    )
    
    print(f"MAPTA LiteLLM — Model: {MAPTA_MODEL}, Sandbox Model: {MAPTA_SANDBOX_MODEL}")
    thinking_status = f"reasoning_effort={MAPTA_REASONING_EFFORT}"
    if MAPTA_REASONING_EFFORT != "none":
        thinking_status += f", thinking_budget={MAPTA_THINKING_BUDGET_TOKENS}"
    print(f"Extended Thinking: {thinking_status}")
    
    system_prompt = os.getenv("SYSTEM_PROMPT", "SYSTEM_PROMPT_REDACTED")

    targets_file = "targets.txt"
    
    if os.path.exists(targets_file):
        print(f"Found targets file: {targets_file}")
        
        targets = read_targets_from_file(targets_file)
        
        if not targets:
            print("No valid targets found in targets.txt file. Falling back to single target mode.")
        else:
            print(f"Found {len(targets)} targets to scan")
            
            base_user_prompt = "I need you to do a full vulnerability scan of {target_url}, you must critically analyse the code and identify every single vulnerability, for identified vulnerabilities a PoC must be provided, focus on critical vulnerabilities, i m only insterested in real world vulnerabilities, not theoretical ones"
            
            results = asyncio.run(run_parallel_scans(targets, system_prompt, base_user_prompt))
            
            print("\nAll scans completed!")
            sys.exit(0)
