# MAPTA — Multi-Agent Penetration Testing Assistant

> **Fork: LiteLLM Edition** — Model-agnostic patch supporting OpenAI, Anthropic, AWS Bedrock, Azure, Ollama, and any LiteLLM-compatible provider.

## What is MAPTA?

MAPTA is an autonomous multi-agent penetration testing framework that achieved **76.9% on the XBOW benchmark** (104 web vulnerability challenges). It uses a multi-agent architecture where a main orchestrator agent delegates work to specialized sandbox and validator agents to discover, exploit, and validate real-world web vulnerabilities.

Originally developed as an academic research project using the OpenAI Responses API with hardcoded `gpt-5`, this fork replaces the entire LLM integration layer with [LiteLLM](https://github.com/BerriAI/litellm) to support **100+ LLM providers** through a single, unified interface.

### Published Benchmark Results

| Vulnerability Category | Solve Rate | Notes |
|---|---|---|
| **SSRF** | 100% | Perfect score |
| **Misconfiguration** | 100% | Perfect score |
| **XSS** | ~85% | Strong |
| **SQL Injection** | ~70% | Struggles with blind SQLi |
| **Overall XBOW** | **76.9%** | 80/104 challenges |
| **Median cost per challenge** | **$0.073** | Extremely efficient |

---

## What Changed in This Fork

### The Problem

The original MAPTA was locked to OpenAI's proprietary Responses API (`/v1/responses`) with hardcoded `gpt-5`. This meant:

- No way to use Anthropic Claude, AWS Bedrock, or local models
- No extended thinking support beyond OpenAI's `reasoning={"effort":"high"}`
- Vendor lock-in to a single provider

### The Solution

Complete rewrite of the LLM integration layer from OpenAI Responses API to the standard Chat Completions API via LiteLLM.

### Changes at a Glance

| Aspect | Original | This Fork |
|---|---|---|
| **LLM Client** | `AsyncOpenAI().responses.create()` | `litellm.acompletion()` |
| **Model** | Hardcoded `gpt-5` | Configurable via `MAPTA_MODEL` env var |
| **Providers** | OpenAI only | OpenAI, Anthropic, Bedrock, Azure, Ollama, vLLM, 100+ more |
| **Message Format** | Responses API (`input`, `developer` role) | Chat Completions (`messages`, `system` role) |
| **Tool Schema** | `{"type":"function","name":...}` | `{"type":"function","function":{"name":...}}` |
| **Tool Results** | `{"type":"function_call_output","call_id":...}` | `{"role":"tool","tool_call_id":...}` |
| **Response Parsing** | `response.output` → `item.type=="function_call"` | `response.choices[0].message.tool_calls` |
| **Extended Thinking** | OpenAI-only `reasoning={"effort":"high"}` | Universal `reasoning_effort` + Anthropic native `thinking` |
| **Thinking + Tools** | Not handled | `thinking_blocks` preserved in conversation history |

### All 3 Agent Loops Patched

MAPTA runs three nested agent loops, and all three were converted:

1. **Main Orchestrator** — coordinates the overall scan, delegates to sandbox/validator agents
2. **Sandbox Agent** — executes commands and Python code inside an isolated Docker container
3. **Validator Agent** — reproduces and validates Proof-of-Concept exploits

Each agent loop independently calls LiteLLM with its own model configuration, enabling cost optimization (e.g., use a cheaper model for sandbox execution, a more capable model for the main orchestrator).

---

## Installation

### Prerequisites

- Python 3.10+
- Docker (for sandbox execution)
- API key for at least one LLM provider

### Setup

```bash
# Clone this fork
git clone https://github.com/YOUR_ORG/sentinel-mapta.git
cd sentinel-mapta

# Install dependencies
pip install litellm>=1.61.20 httpx aiohttp

# Set your API key (pick one provider)
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
# or
export AWS_ACCESS_KEY_ID=... && export AWS_SECRET_ACCESS_KEY=... && export AWS_REGION_NAME=us-east-1
```

---

## Configuration

All configuration is done via environment variables. No code changes needed to switch providers.

### Model Selection

```bash
# Main orchestrator model — the "brain" that plans and coordinates
export MAPTA_MODEL=anthropic/claude-sonnet-4-20250514

# Sandbox/validator model — executes commands, can be cheaper
# Falls back to MAPTA_MODEL if not set
export MAPTA_SANDBOX_MODEL=anthropic/claude-haiku-4-5-20251001
```

### Provider Examples

```bash
# ── Anthropic (Direct API) ──
export MAPTA_MODEL=anthropic/claude-sonnet-4-20250514
export ANTHROPIC_API_KEY=sk-ant-...

# ── AWS Bedrock (3 authentication methods) ──
#
# Method 1: Standard IAM Credentials (most common)
#   Uses AWS access key + secret key — works with any IAM user/role
export MAPTA_MODEL=bedrock/anthropic.claude-sonnet-4-20250514-v1:0
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=wJalr...
export AWS_REGION_NAME=us-east-1
#
# Method 2: Bedrock API Key (ABSK... format)
#   These are the newer Bedrock-specific keys that start with "ABSK" followed
#   by base64. They are bearer tokens — NOT standard IAM keys.
#   ⚠️ LiteLLM's bedrock/ provider uses the IAM credential chain by default.
#   To use Bedrock API Keys, pass them via extra_headers as a bearer token:
export MAPTA_MODEL=bedrock/anthropic.claude-sonnet-4-20250514-v1:0
export AWS_REGION_NAME=us-east-1
export BEDROCK_API_KEY="ABSKQmVkcm9ja0FQ..."
#   Then in your code or wrapper, pass: extra_headers={"Authorization": f"Bearer {BEDROCK_API_KEY}"}
#   See "Bedrock API Key Setup" section below for integration details.
#
# Method 3: AWS SSO / Profile-based
#   Uses your configured AWS CLI profile (run `aws sso login` first)
export MAPTA_MODEL=bedrock/anthropic.claude-sonnet-4-20250514-v1:0
export AWS_PROFILE=your-sso-profile
export AWS_REGION_NAME=us-east-1

# ── OpenAI ──
export MAPTA_MODEL=openai/gpt-4o
export OPENAI_API_KEY=sk-...

# ── OpenAI Reasoning Models ──
export MAPTA_MODEL=openai/o3
export OPENAI_API_KEY=sk-...

# ── Google Gemini ──
export MAPTA_MODEL=gemini/gemini-2.5-pro
export GEMINI_API_KEY=...

# ── DeepSeek ──
export MAPTA_MODEL=deepseek/deepseek-chat
export DEEPSEEK_API_KEY=...

# ── Local Models (Ollama) ──
export MAPTA_MODEL=ollama/llama3
# No API key needed — Ollama runs locally

# ── Local Models (vLLM) ──
export MAPTA_MODEL=openai/my-model
export OPENAI_API_BASE=http://localhost:8000
```

### Bedrock API Key Setup (ABSK... Keys)

AWS Bedrock API Keys (`ABSK...` format) are a newer authentication method that works as bearer tokens instead of standard IAM SigV4 signing. If you have one of these keys, here's how to use it with MAPTA:

**What the key looks like:**
```
ABSKQmVkcm9ja0FQSUtleS1maDUzLWF0LTcwMjQyMjg5MzI3MDphc1VSZEVmemtOUXhYTU8r...
```

**How to use it:**

The `BEDROCK_API_KEY` env var is read by the patched `_build_reasoning_kwargs()` function and passed as a bearer token header. Add this to your environment:

```bash
export MAPTA_MODEL=bedrock/anthropic.claude-sonnet-4-20250514-v1:0
export AWS_REGION_NAME=us-east-1
export BEDROCK_API_KEY="ABSKQmVkcm9ja0FQ..."

# IMPORTANT: Do NOT set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY when using
# Bedrock API Keys — they will conflict. Use one auth method, not both.
```

The code passes this as an `Authorization: Bearer` header via LiteLLM's `extra_headers` parameter. See the `_get_bedrock_extra_headers()` helper in `main_litellm.py`.

**Key types comparison:**

| Key Type | Format | Auth Method | Lifespan |
|---|---|---|---|
| IAM Access Key | `AKIA...` + secret | SigV4 signing | Long-lived or rotated |
| Bedrock API Key (long-term) | `ABSK...` (132 chars) | Bearer token | Configurable expiry |
| Bedrock API Key (short-term) | `bedrock-api-key-...` (1000+ chars) | Bearer token | Short-lived (presigned URL) |

> **Note:** Standard IAM credentials (Method 1) are the most widely tested path with LiteLLM. Bedrock API Keys are newer and may require LiteLLM version 1.61.20+ for full support.

### Extended Thinking

Extended thinking enables deeper reasoning before the model responds. LiteLLM automatically translates this into each provider's native format.

```bash
# Enable extended thinking (default: none = disabled)
export MAPTA_REASONING_EFFORT=high    # none | low | medium | high

# Anthropic-specific: set thinking token budget (default: 8192, min: 1024)
export MAPTA_THINKING_BUDGET_TOKENS=10000
```

**How it translates per provider:**

| `MAPTA_REASONING_EFFORT` | OpenAI (o3/gpt-5) | Anthropic (Claude) | Gemini | DeepSeek | Other |
|---|---|---|---|---|---|
| `none` | Normal | Normal | Normal | Normal | Normal |
| `low` | `reasoning={"effort":"low"}` | `thinking={"type":"enabled","budget_tokens":N}` | `thinking_budget` | Native reasoning | Silently ignored |
| `medium` | `reasoning={"effort":"medium"}` | `thinking={"type":"enabled","budget_tokens":N}` | `thinking_budget` | Native reasoning | Silently ignored |
| `high` | `reasoning={"effort":"high"}` | `thinking={"type":"enabled","budget_tokens":N}` | `thinking_budget` | Native reasoning | Silently ignored |

**Thinking + Tool Calling Safety:** Anthropic requires `thinking_blocks` to be preserved in conversation history during multi-turn tool use. This fork handles this automatically through two mechanisms:

1. `_build_assistant_message()` preserves `thinking_blocks` from each response
2. `litellm.modify_params = True` auto-drops thinking params on turns where blocks are missing (prevents the common Anthropic crash: *"Expected `thinking` or `redacted_thinking`, but found `text`"*)

### Slack Integration (Optional)

```bash
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
export SLACK_CHANNEL=#security-alerts
```

When configured, MAPTA sends real-time vulnerability alerts and scan summaries to Slack.

### Sandbox Configuration

MAPTA executes code inside an isolated sandbox. Provide a factory function:

```bash
# Format: "module_path:function_name"
# The function must return an object with:
#   .files.write(path, content)
#   .commands.run(cmd, timeout=..., user=...)
#   .set_timeout(ms)  (optional)
#   .kill()           (optional)
export SANDBOX_FACTORY=your_sandbox_module:create_sandbox
```

---

## Usage

### Single Target Scan

Set the `SYSTEM_PROMPT` env var with your scanning instructions, then provide targets in `targets.txt`:

```bash
# Set your system prompt (or use the default)
export SYSTEM_PROMPT="You are an expert penetration tester..."

# Create targets file
echo "https://target1.example.com" > targets.txt
echo "https://target2.example.com" >> targets.txt

# Run
python main_litellm.py
```

MAPTA will:
1. Read all targets from `targets.txt`
2. Launch parallel scans (one per target, each with its own sandbox)
3. Save results as markdown files (one per target)
4. Save usage/cost data as JSON files
5. Send Slack alerts for discovered vulnerabilities (if configured)

### Startup Output

```
MAPTA LiteLLM — Model: anthropic/claude-sonnet-4-20250514, Sandbox Model: anthropic/claude-haiku-4-5-20251001
Extended Thinking: reasoning_effort=high, thinking_budget=10000
Found targets file: targets.txt
Found 2 targets to scan
Using model: anthropic/claude-sonnet-4-20250514 (sandbox: anthropic/claude-haiku-4-5-20251001)
Starting parallel scans for 2 targets...
```

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  MAPTA Runner                    │
│         (main_litellm.py — entry point)          │
│    Reads targets.txt, launches parallel scans    │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │    Main Orchestrator     │
          │   (MAPTA_MODEL)          │
          │                          │
          │  Plans attack strategy   │
          │  Delegates to sub-agents │
          │  Reports findings        │
          │  Sends Slack alerts      │
          └────┬──────────┬─────────┘
               │          │
    ┌──────────▼──┐  ┌────▼──────────┐
    │ Sandbox Agent│  │Validator Agent│
    │(SANDBOX_MODEL│  │(SANDBOX_MODEL)│
    │              │  │               │
    │ Runs bash    │  │ Reproduces    │
    │ Runs Python  │  │ PoCs in       │
    │ In isolated  │  │ sandbox       │
    │ Docker       │  │ Confirms      │
    │ container    │  │ exploits      │
    └──────┬───────┘  └───────┬───────┘
           │                  │
    ┌──────▼──────────────────▼──────┐
    │     Isolated Docker Sandbox     │
    │   (per-scan, auto-destroyed)    │
    │                                 │
    │  sandbox_run_command(cmd)       │
    │  sandbox_run_python(code)       │
    └─────────────────────────────────┘
```

### Tool Registry

The main orchestrator has access to these tools:

| Tool | Description |
|---|---|
| `sandbox_agent` | Nested agent that executes bash/Python in the sandbox |
| `validator_agent` | Nested agent that validates PoCs for real-world impact |
| `get_registered_emails` | Lists temp email accounts for testing auth flows |
| `list_account_messages` | Reads messages from temp email accounts |
| `get_message_by_id` | Fetches specific email content |
| `send_slack_alert` | Sends vulnerability alert to Slack |
| `send_slack_summary` | Sends scan summary to Slack |

The sandbox and validator agents only have access to `sandbox_run_command` and `sandbox_run_python` — preventing recursive nesting.

---

## Cost Optimization

MAPTA's original median cost was **$0.073 per challenge** on OpenAI. Here are strategies to optimize cost on different providers:

### Use Split Models

Run the orchestrator on a capable model and sandbox execution on a cheaper one:

```bash
export MAPTA_MODEL=anthropic/claude-sonnet-4-20250514       # Smart orchestrator
export MAPTA_SANDBOX_MODEL=anthropic/claude-haiku-4-5-20251001  # Cheap executor
```

### Tune Thinking Budget

Extended thinking costs tokens. Start with the minimum and increase if solve rates drop:

```bash
export MAPTA_REASONING_EFFORT=low           # Start low
export MAPTA_THINKING_BUDGET_TOKENS=2048    # Minimum practical budget
```

### Use Local Models for Sandbox

For sandbox execution (running bash commands, Python scripts), a local model is often sufficient:

```bash
export MAPTA_MODEL=anthropic/claude-sonnet-4-20250514   # Cloud orchestrator
export MAPTA_SANDBOX_MODEL=ollama/llama3                 # Free local executor
```

---

## File Structure

```
sentinel-mapta/
├── main_litellm.py          # Patched main file (this fork)
├── main.py                  # Original OpenAI-only version (reference)
├── function_tool.py         # Tool decorator (unchanged)
├── analyze_logs.py          # Log analysis utility
├── targets.txt              # Target URLs (one per line)
├── requirements_litellm.txt # Additional dependencies for this fork
└── ctf-logs/                # XBOW benchmark run logs (104 challenges)
    ├── analysis_output/     # Aggregated analysis results
    ├── XBOW 1_*.log
    ├── XBOW 2_*.log
    └── ...
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `MAPTA_MODEL` | No | `gpt-4o` | Main orchestrator model (LiteLLM format) |
| `MAPTA_SANDBOX_MODEL` | No | Same as `MAPTA_MODEL` | Sandbox/validator model |
| `MAPTA_REASONING_EFFORT` | No | `none` | Extended thinking: `none`, `low`, `medium`, `high` |
| `MAPTA_THINKING_BUDGET_TOKENS` | No | `8192` | Anthropic thinking token budget (min 1024) |
| `SYSTEM_PROMPT` | No | Default prompt | System prompt for the main orchestrator |
| `SANDBOX_SYSTEM_PROMPT` | No | Default prompt | System prompt for the sandbox agent |
| `VALIDATOR_SYSTEM_PROMPT` | No | Default prompt | System prompt for the validator agent |
| `SANDBOX_FACTORY` | No | None | Sandbox factory: `module:function` |
| `SLACK_WEBHOOK_URL` | No | None | Slack webhook for alerts |
| `SLACK_CHANNEL` | No | `#security-alerts` | Slack channel name |
| `OPENAI_API_KEY` | If using OpenAI | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | If using Anthropic | — | Anthropic API key |
| `AWS_ACCESS_KEY_ID` | If using Bedrock (IAM) | — | AWS IAM access key |
| `AWS_SECRET_ACCESS_KEY` | If using Bedrock (IAM) | — | AWS IAM secret key |
| `AWS_REGION_NAME` | If using Bedrock | — | AWS region (e.g. `us-east-1`) |
| `AWS_PROFILE` | If using Bedrock (SSO) | — | AWS CLI profile name |
| `BEDROCK_API_KEY` | If using Bedrock API Key | — | Bedrock API key (`ABSK...` format) |
| `GEMINI_API_KEY` | If using Gemini | — | Google Gemini API key |

---

## Context: Sentinel Engine Integration

This MAPTA fork is part of the **Sentinel Engine** project — an autonomous pentesting orchestrator that cascades multiple AI agents in a tiered architecture to maximize vulnerability discovery.

### Sentinel Web Module — Cascade Order

```
Tier 1: Cyber-AutoAgent + Sonnet 4   (85% XBOW — highest score)
    ↓ fails
Tier 2: Cyber-AutoAgent + Opus       (same agent, better model)
    ↓ fails
Tier 3: Deadend CLI                   (78% — different approach, solves blind SQLi)
    ↓ fails
Tier 4: MAPTA (this tool)             (76.9% — cheapest, different architecture)
```

MAPTA serves as Tier 4 in the cascade. Its key strength is cost efficiency ($0.073/challenge) and a different multi-agent architecture that catches vulnerabilities the other agents miss — particularly SSRF and misconfiguration issues where it scores 100%.

The LiteLLM patch is essential for Sentinel Engine integration because it allows the orchestrator to route any provider/model to any agent through a unified interface.

---

## License

MIT — Same as the original MAPTA.

AGPL-3.0 components (Deadend CLI) are used separately in the Sentinel cascade and do not affect this repository.

---

## Credits

- **Original MAPTA**: [arthurgervais/mapta](https://github.com/arthurgervais/mapta)
- **LiteLLM**: [BerriAI/litellm](https://github.com/BerriAI/litellm)
- **LiteLLM Patch**: Sentinel Engine project (TechFirio LLC)
