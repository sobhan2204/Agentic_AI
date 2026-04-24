"""
github.py  —  Unified Developer MCP Tool Server
================================================
Exposes 4 MCP tools in a single file:

  1. github_tool        — GitHub API: search repos, read files, list/create issues, get READMEs
  2. code_executor      — Sandbox: run Python code safely, capture stdout/stderr/errors
  3. code_writer        — AI-style structured code generation with explanations
  4. github_run_review  — Integrated: fetch code from GitHub → execute → analyse mistakes/improvements

Register in TOOL_REGISTRY (client.py) as:
    "github": {"file": "github.py", ...}

Requires in .env:
    GITHUB_TOKEN=ghp_xxxxxxxxxxxx   (optional but strongly recommended — avoids 60 req/hr rate limit)

Run standalone for testing:
    python github.py
"""

import sys
import os
import ast
import io
import re
import json
import base64
import textwrap
import traceback
import subprocess
import tempfile
import contextlib
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

# ── MCP server bootstrap ───────────────────────────────────────────────────────
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError("Install mcp: pip install mcp")

try:
    import requests
except ImportError:
    raise ImportError("Install requests: pip install requests")

load_dotenv()

mcp = FastMCP("developer_tools")

GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")
GITHUB_API    = "https://api.github.com"
EXEC_TIMEOUT  = 10   # seconds for code execution sandbox


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _gh_headers() -> dict:
    """Build GitHub API request headers. Auth token is optional but recommended."""
    headers = {
        "Accept":     "application/vnd.github+json",
        "User-Agent": "Sobhan-MCP-Agent/1.0",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def _gh_get(url: str, params: dict = None) -> dict:
    """
    Make a GitHub API GET request.
    Returns a dict with keys: ok (bool), data (any), status (int), error (str).
    """
    try:
        resp = requests.get(url, headers=_gh_headers(), params=params or {}, timeout=15)
        if resp.status_code == 200:
            return {"ok": True, "data": resp.json(), "status": 200, "error": ""}
        else:
            try:
                msg = resp.json().get("message", resp.text[:200])
            except Exception:
                msg = resp.text[:200]
            return {"ok": False, "data": None, "status": resp.status_code, "error": msg}
    except requests.exceptions.Timeout:
        return {"ok": False, "data": None, "status": 0, "error": "Request timed out"}
    except requests.exceptions.ConnectionError as e:
        return {"ok": False, "data": None, "status": 0, "error": f"Connection error: {str(e)[:120]}"}


def _gh_post(url: str, payload: dict) -> dict:
    """Make a GitHub API POST request (used for creating issues)."""
    try:
        resp = requests.post(url, headers=_gh_headers(), json=payload, timeout=15)
        if resp.status_code in (200, 201):
            return {"ok": True, "data": resp.json(), "status": resp.status_code, "error": ""}
        else:
            try:
                msg = resp.json().get("message", resp.text[:200])
            except Exception:
                msg = resp.text[:200]
            return {"ok": False, "data": None, "status": resp.status_code, "error": msg}
    except Exception as e:
        return {"ok": False, "data": None, "status": 0, "error": str(e)[:200]}


def _decode_content(content_b64: str) -> str:
    """Decode base64-encoded GitHub file content."""
    try:
        return base64.b64decode(content_b64).decode("utf-8", errors="replace")
    except Exception:
        return "[Could not decode file content]"


def _truncate(text: str, max_chars: int = 3000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n... [truncated — {len(text) - max_chars} more chars]"


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 1 — GITHUB API
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def github_tool(
    action: str,
    owner: str = "",
    repo: str = "",
    query: str = "",
    path: str = "",
    branch: str = "main",
    issue_title: str = "",
    issue_body: str = "",
    issue_number: int = 0,
) -> str:
    """
    Interact with GitHub repositories.

    Actions:
      search_repos   — Search GitHub for repositories matching `query`
      get_readme     — Fetch the README of `owner/repo`
      read_file      — Read a specific file at `path` in `owner/repo` (on `branch`)
      list_issues    — List open issues in `owner/repo`
      get_issue      — Get a single issue by `issue_number` in `owner/repo`
      create_issue   — Create a new issue in `owner/repo` with `issue_title` and `issue_body`
      list_files     — List files/folders at `path` in `owner/repo` (on `branch`)
      repo_info      — Get metadata about `owner/repo` (stars, language, description, etc.)

    Args:
        action:        One of the actions listed above.
        owner:         GitHub username or organisation (e.g. "torvalds").
        repo:          Repository name (e.g. "linux").
        query:         Search query (used by search_repos).
        path:          File or folder path inside the repo (used by read_file, list_files).
        branch:        Branch name. Defaults to "main".
        issue_title:   Title for new issue (used by create_issue).
        issue_body:    Body/description for new issue (used by create_issue).
        issue_number:  Issue number (used by get_issue).
    """

    action = action.strip().lower()

    # ── search_repos ──────────────────────────────────────────────────────────
    if action == "search_repos":
        if not query:
            return "Error: 'query' is required for search_repos."
        result = _gh_get(f"{GITHUB_API}/search/repositories", params={"q": query, "per_page": 5, "sort": "stars"})
        if not result["ok"]:
            return f"GitHub search failed ({result['status']}): {result['error']}"
        items = result["data"].get("items", [])
        if not items:
            return f"No repositories found for query: '{query}'"
        lines = [f"Top {len(items)} results for '{query}':\n"]
        for r in items:
            lines.append(
                f"• {r['full_name']}  ⭐ {r.get('stargazers_count', 0):,}\n"
                f"  {r.get('description') or 'No description'}\n"
                f"  Language: {r.get('language') or 'N/A'}  |  "
                f"URL: {r.get('html_url', '')}"
            )
        return "\n".join(lines)

    # ── repo_info ─────────────────────────────────────────────────────────────
    if action == "repo_info":
        if not owner or not repo:
            return "Error: 'owner' and 'repo' are required for repo_info."
        result = _gh_get(f"{GITHUB_API}/repos/{owner}/{repo}")
        if not result["ok"]:
            return f"Could not fetch repo info ({result['status']}): {result['error']}"
        d = result["data"]
        return (
            f"Repository: {d['full_name']}\n"
            f"Description: {d.get('description') or 'None'}\n"
            f"Language: {d.get('language') or 'N/A'}\n"
            f"Stars: {d.get('stargazers_count', 0):,}  |  Forks: {d.get('forks_count', 0):,}\n"
            f"Open issues: {d.get('open_issues_count', 0)}\n"
            f"Default branch: {d.get('default_branch', 'main')}\n"
            f"URL: {d.get('html_url', '')}\n"
            f"Created: {d.get('created_at', 'N/A')[:10]}  |  "
            f"Last pushed: {d.get('pushed_at', 'N/A')[:10]}"
        )

    # ── get_readme ────────────────────────────────────────────────────────────
    if action == "get_readme":
        if not owner or not repo:
            return "Error: 'owner' and 'repo' are required for get_readme."
        result = _gh_get(f"{GITHUB_API}/repos/{owner}/{repo}/readme")
        if not result["ok"]:
            return f"Could not fetch README ({result['status']}): {result['error']}"
        content = _decode_content(result["data"].get("content", ""))
        return _truncate(f"README for {owner}/{repo}:\n\n{content}")

    # ── read_file ─────────────────────────────────────────────────────────────
    if action == "read_file":
        if not owner or not repo or not path:
            return "Error: 'owner', 'repo', and 'path' are required for read_file."
        result = _gh_get(
            f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path.lstrip('/')}",
            params={"ref": branch},
        )
        if not result["ok"]:
            return f"Could not read file ({result['status']}): {result['error']}"
        data = result["data"]
        if isinstance(data, list):
            return f"'{path}' is a directory. Use list_files to see its contents."
        content = _decode_content(data.get("content", ""))
        return _truncate(
            f"File: {owner}/{repo}/{path}  (branch: {branch})\n"
            f"Size: {data.get('size', 0):,} bytes\n\n"
            f"{content}"
        )

    # ── list_files ────────────────────────────────────────────────────────────
    if action == "list_files":
        if not owner or not repo:
            return "Error: 'owner' and 'repo' are required for list_files."
        url_path = path.lstrip("/") if path else ""
        result = _gh_get(
            f"{GITHUB_API}/repos/{owner}/{repo}/contents/{url_path}",
            params={"ref": branch},
        )
        if not result["ok"]:
            return f"Could not list files ({result['status']}): {result['error']}"
        items = result["data"]
        if not isinstance(items, list):
            return "This path points to a file, not a directory. Use read_file to read it."
        dirs  = [i["name"] + "/" for i in items if i["type"] == "dir"]
        files = [i["name"] for i in items if i["type"] == "file"]
        return (
            f"Contents of {owner}/{repo}/{path or ''}  (branch: {branch}):\n\n"
            f"Folders ({len(dirs)}): {', '.join(dirs) or 'none'}\n"
            f"Files   ({len(files)}): {', '.join(files) or 'none'}"
        )

    # ── list_issues ───────────────────────────────────────────────────────────
    if action == "list_issues":
        if not owner or not repo:
            return "Error: 'owner' and 'repo' are required for list_issues."
        result = _gh_get(
            f"{GITHUB_API}/repos/{owner}/{repo}/issues",
            params={"state": "open", "per_page": 10},
        )
        if not result["ok"]:
            return f"Could not fetch issues ({result['status']}): {result['error']}"
        issues = [i for i in result["data"] if "pull_request" not in i]
        if not issues:
            return f"No open issues found in {owner}/{repo}."
        lines = [f"Open issues in {owner}/{repo}:\n"]
        for i in issues:
            labels = ", ".join(l["name"] for l in i.get("labels", [])) or "none"
            lines.append(
                f"  #{i['number']}  {i['title']}\n"
                f"    Labels: {labels}  |  Created: {i['created_at'][:10]}\n"
                f"    URL: {i['html_url']}"
            )
        return "\n".join(lines)

    # ── get_issue ─────────────────────────────────────────────────────────────
    if action == "get_issue":
        if not owner or not repo or not issue_number:
            return "Error: 'owner', 'repo', and 'issue_number' are required for get_issue."
        result = _gh_get(f"{GITHUB_API}/repos/{owner}/{repo}/issues/{issue_number}")
        if not result["ok"]:
            return f"Could not fetch issue #{issue_number} ({result['status']}): {result['error']}"
        i = result["data"]
        labels = ", ".join(l["name"] for l in i.get("labels", [])) or "none"
        return (
            f"Issue #{i['number']}: {i['title']}\n"
            f"State: {i['state']}  |  Labels: {labels}\n"
            f"Author: {i['user']['login']}  |  Created: {i['created_at'][:10]}\n"
            f"URL: {i['html_url']}\n\n"
            f"Body:\n{_truncate(i.get('body') or 'No description provided.')}"
        )

    # ── create_issue ──────────────────────────────────────────────────────────
    if action == "create_issue":
        if not owner or not repo or not issue_title:
            return "Error: 'owner', 'repo', and 'issue_title' are required for create_issue."
        payload = {"title": issue_title, "body": issue_body or ""}
        result  = _gh_post(f"{GITHUB_API}/repos/{owner}/{repo}/issues", payload)
        if not result["ok"]:
            return f"Could not create issue ({result['status']}): {result['error']}"
        i = result["data"]
        return (
            f"Issue created successfully!\n"
            f"  #{i['number']}: {i['title']}\n"
            f"  URL: {i['html_url']}"
        )

    return (
        f"Unknown action: '{action}'. "
        "Valid actions: search_repos, repo_info, get_readme, read_file, "
        "list_files, list_issues, get_issue, create_issue."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 2 — CODE EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

def _run_python_safe(code: str, timeout: int = EXEC_TIMEOUT) -> dict:
    """
    Execute Python code in a subprocess sandbox.
    Returns dict: stdout, stderr, error, exit_code, timed_out.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return {
            "stdout":    result.stdout,
            "stderr":    result.stderr,
            "error":     "",
            "exit_code": result.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout":    "",
            "stderr":    "",
            "error":     f"Execution timed out after {timeout} seconds.",
            "exit_code": -1,
            "timed_out": True,
        }
    except Exception as e:
        return {
            "stdout":    "",
            "stderr":    "",
            "error":     str(e),
            "exit_code": -1,
            "timed_out": False,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@mcp.tool()
def code_executor(code: str, language: str = "python") -> str:
    """
    Execute a code snippet and return its output, errors, and exit code.

    Currently supports Python only. The code runs in an isolated subprocess
    with a 10-second timeout so infinite loops cannot hang the agent.

    Args:
        code:     The source code to execute.
        language: Programming language. Currently only "python" is supported.
    """
    language = (language or "python").strip().lower()

    if language != "python":
        return (
            f"Language '{language}' is not supported yet. "
            "Only Python is currently available in the sandbox."
        )

    if not code or not code.strip():
        return "Error: No code provided to execute."

    result = _run_python_safe(code.strip())

    parts = []

    if result["timed_out"]:
        parts.append("⏱️  TIMEOUT: Code exceeded the 10-second execution limit.")
        parts.append("Check for infinite loops or very slow operations.")
        return "\n".join(parts)

    if result["stdout"]:
        parts.append(f"📤 Output:\n{_truncate(result['stdout'], 2000)}")

    if result["stderr"]:
        parts.append(f"⚠️  Stderr:\n{_truncate(result['stderr'], 1000)}")

    if result["error"]:
        parts.append(f"❌ Error: {result['error']}")

    parts.append(f"Exit code: {result['exit_code']}")

    if result["exit_code"] == 0 and not result["stdout"] and not result["stderr"]:
        parts.append("✅ Code ran successfully with no output.")

    return "\n\n".join(parts) if parts else "No output produced."


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 3 — CODE WRITER
# ═══════════════════════════════════════════════════════════════════════════════

def _static_analyse(code: str) -> list:
    """
    Run static analysis on Python code using the ast module.
    Returns a list of warning strings.
    """
    warnings = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError at line {e.lineno}: {e.msg}"]

    for node in ast.walk(tree):
        # Bare except
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            warnings.append(f"Line {node.lineno}: Bare 'except' clause — consider catching a specific exception.")
        # Missing return type hints on functions
        if isinstance(node, ast.FunctionDef) and node.returns is None:
            warnings.append(f"Line {node.lineno}: Function '{node.name}' has no return type annotation.")
        # Missing docstring on functions
        if isinstance(node, ast.FunctionDef):
            if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)):
                warnings.append(f"Line {node.lineno}: Function '{node.name}' has no docstring.")
        # Global variable use
        if isinstance(node, ast.Global):
            warnings.append(f"Line {node.lineno}: 'global' keyword used — consider refactoring to avoid mutable global state.")
        # print() left in code
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                warnings.append(f"Line {node.lineno}: 'print()' found — use logging instead for production code.")

    return warnings


@mcp.tool()
def code_writer(
    task: str,
    language: str = "python",
    style: str = "clean",
    run_after_writing: bool = True,
) -> str:
    """
    Generate well-structured code for a given task, with explanations and static analysis.

    This tool writes code, explains what it does, analyses it for common issues,
    and optionally executes it to verify correctness.

    Args:
        task:               Natural-language description of what the code should do.
                            Example: "Write a function that reverses a string and handles edge cases."
        language:           Target programming language. Defaults to "python".
        style:              Code style preference: "clean", "verbose" (with lots of comments),
                            or "minimal" (as short as possible). Defaults to "clean".
        run_after_writing:  If True, execute the generated code and include output. Defaults to True.
    """
    language = (language or "python").strip().lower()
    style    = (style or "clean").strip().lower()

    if language != "python":
        return f"Code generation currently supports Python only. Language '{language}' is not available."

    if not task or not task.strip():
        return "Error: 'task' cannot be empty. Describe what the code should do."

    # ── Build the code based on the task (template-driven generation) ─────────
    # NOTE: In the real agent, the LLM handles actual code generation.
    # This function structures the output format and runs static analysis + execution.
    # Here we produce a well-formatted stub so the MCP tool is fully functional.

    task_clean = task.strip()

    if style == "verbose":
        comment_style = "# Verbose mode: detailed comments throughout"
    elif style == "minimal":
        comment_style = "# Minimal style"
    else:
        comment_style = "# Clean, readable Python"

    # Produce a template code block that the agent will fill in via its LLM call
    generated_code = textwrap.dedent(f"""\
        {comment_style}
        # Task: {task_clean}
        # Generated by Sobhan_AI Developer Tools
        # Language: {language}

        def main():
            \"\"\"Entry point — implement the task logic here.\"\"\"
            # TODO: implement — '{task_clean}'
            pass

        if __name__ == "__main__":
            main()
    """)

    # ── Static analysis ───────────────────────────────────────────────────────
    warnings = _static_analyse(generated_code)

    parts = [
        f"📝 Code generated for task:\n   '{task_clean}'\n",
        f"```python\n{generated_code}\n```",
    ]

    if warnings:
        parts.append("🔍 Static Analysis:\n" + "\n".join(f"  • {w}" for w in warnings))
    else:
        parts.append("✅ Static Analysis: No issues found.")

    # ── Optionally execute ────────────────────────────────────────────────────
    if run_after_writing:
        exec_result = _run_python_safe(generated_code)
        if exec_result["timed_out"]:
            parts.append("⏱️  Execution: Timed out.")
        elif exec_result["exit_code"] == 0:
            output = exec_result["stdout"] or "(no output)"
            parts.append(f"▶️  Execution: Success\n{output}")
        else:
            parts.append(
                f"❌ Execution failed (exit {exec_result['exit_code']}):\n"
                f"{exec_result['stderr'] or exec_result['error']}"
            )

    parts.append(
        "💡 Tip: Paste your own implementation in place of the stub, "
        "then call code_executor to run and verify it."
    )

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 4 — GITHUB FETCH → EXECUTE → REVIEW  (integrated)
# ═══════════════════════════════════════════════════════════════════════════════

def _review_code(code: str, exec_result: dict) -> str:
    """
    Analyse code and execution results to produce a structured review:
    mistakes found, improvements suggested, and a verdict.
    """
    lines = code.splitlines()
    issues      = []
    suggestions = []

    # ── Static analysis ───────────────────────────────────────────────────────
    static_warnings = _static_analyse(code)
    for w in static_warnings:
        issues.append(f"[Static] {w}")

    # ── Execution analysis ────────────────────────────────────────────────────
    if exec_result["timed_out"]:
        issues.append("[Runtime] Code execution timed out — likely an infinite loop or blocking I/O.")
        suggestions.append("Add a termination condition or use generators for large data processing.")

    elif exec_result["exit_code"] != 0:
        stderr = exec_result.get("stderr", "") or exec_result.get("error", "")
        # Parse traceback for specific error type
        error_lines = [l for l in stderr.splitlines() if l.strip()]
        if error_lines:
            # Find the actual error line
            for line in reversed(error_lines):
                if re.match(r"[A-Z][a-zA-Z]+Error", line) or re.match(r"[A-Z][a-zA-Z]+Exception", line):
                    issues.append(f"[Runtime] {line.strip()}")
                    break
            else:
                issues.append(f"[Runtime] {error_lines[-1].strip()}")

        # Give targeted suggestions based on error type
        if "NameError" in stderr:
            suggestions.append("A variable or function is used before being defined. Check spelling and scope.")
        if "TypeError" in stderr:
            suggestions.append("A function received the wrong type of argument. Check input types and function signatures.")
        if "IndexError" in stderr:
            suggestions.append("List/tuple index is out of range. Add bounds checking before accessing elements.")
        if "KeyError" in stderr:
            suggestions.append("Dictionary key does not exist. Use .get(key, default) for safe access.")
        if "ImportError" in stderr or "ModuleNotFoundError" in stderr:
            suggestions.append("A required module is missing. Install it with 'pip install <module_name>'.")
        if "IndentationError" in stderr:
            suggestions.append("Indentation is inconsistent. Use 4 spaces per level throughout.")
        if "RecursionError" in stderr:
            suggestions.append("Maximum recursion depth exceeded. Add a base case or convert recursion to iteration.")
        if "ZeroDivisionError" in stderr:
            suggestions.append("Division by zero detected. Add a check: 'if denominator != 0' before dividing.")
        if "AttributeError" in stderr:
            suggestions.append("Object does not have the expected attribute or method. Verify the object type.")
        if "ValueError" in stderr:
            suggestions.append("A function received an argument of the correct type but invalid value.")

    # ── Code quality checks ───────────────────────────────────────────────────
    code_lower = code.lower()

    if len(lines) > 5 and not any(l.strip().startswith('"""') or l.strip().startswith("'''") for l in lines[:10]):
        suggestions.append("Add a module-level docstring at the top to explain the file's purpose.")

    if "try" not in code_lower and ("open(" in code or "requests." in code or "connect(" in code_lower):
        suggestions.append("File/network operations are not wrapped in try/except — add error handling.")

    if re.search(r'password\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
        issues.append("[Security] Hardcoded password detected. Use environment variables instead.")

    if re.search(r'secret\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
        issues.append("[Security] Hardcoded secret detected. Move to .env or a secrets manager.")

    if "while True" in code and "break" not in code:
        issues.append("[Logic] Infinite loop detected: 'while True' with no 'break' statement.")

    long_lines = [i + 1 for i, l in enumerate(lines) if len(l) > 120]
    if long_lines:
        sample = long_lines[:3]
        suggestions.append(f"Lines {sample} exceed 120 characters. Break them up for readability (PEP 8 recommends ≤ 79).")

    # ── Build review output ───────────────────────────────────────────────────
    verdict = "✅ Code looks good!" if not issues else f"❌ {len(issues)} issue(s) found."

    review_parts = [f"Verdict: {verdict}"]

    if issues:
        review_parts.append("Mistakes / Issues:\n" + "\n".join(f"  {i+1}. {issue}" for i, issue in enumerate(issues)))

    if suggestions:
        review_parts.append("Improvements:\n" + "\n".join(f"  • {s}" for s in suggestions))

    if exec_result["exit_code"] == 0 and exec_result["stdout"]:
        review_parts.append(f"Output:\n{_truncate(exec_result['stdout'], 1000)}")

    return "\n\n".join(review_parts)


@mcp.tool()
def github_run_review(
    owner: str,
    repo: str,
    path: str,
    branch: str = "main",
) -> str:
    """
    Fetch a Python file from GitHub, execute it in the sandbox, then review
    the code for mistakes and suggest improvements.

    This is the integrated tool that combines github_tool + code_executor
    into one end-to-end developer workflow:
      1. Fetch the file from the GitHub repo
      2. Run it in the Python sandbox
      3. Analyse the code + execution results
      4. Return a structured review with mistakes and improvement suggestions

    Args:
        owner:   GitHub username or organisation (e.g. "torvalds").
        repo:    Repository name (e.g. "linux").
        path:    Path to the Python file inside the repo (e.g. "src/utils.py").
        branch:  Branch to read from. Defaults to "main".
    """
    if not owner or not repo or not path:
        return "Error: 'owner', 'repo', and 'path' are all required."

    if not path.endswith(".py"):
        return (
            f"'{path}' does not appear to be a Python file. "
            "github_run_review currently supports .py files only."
        )

    # ── Step 1: Fetch file from GitHub ────────────────────────────────────────
    print(f"[github_run_review] Fetching {owner}/{repo}/{path} @ {branch}")
    gh_result = _gh_get(
        f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path.lstrip('/')}",
        params={"ref": branch},
    )

    if not gh_result["ok"]:
        return (
            f"Could not fetch '{path}' from {owner}/{repo} "
            f"(HTTP {gh_result['status']}): {gh_result['error']}"
        )

    data = gh_result["data"]
    if isinstance(data, list):
        return f"'{path}' is a directory. Please provide a path to a specific .py file."

    code = _decode_content(data.get("content", ""))
    if not code.strip():
        return f"The file '{path}' appears to be empty."

    # ── Step 2: Execute the code ──────────────────────────────────────────────
    print(f"[github_run_review] Executing {len(code)} chars of Python...")
    exec_result = _run_python_safe(code)

    # ── Step 3: Review ────────────────────────────────────────────────────────
    review = _review_code(code, exec_result)

    # ── Step 4: Compose full report ───────────────────────────────────────────
    lines_count = len(code.splitlines())
    header = (
        f"🔬 Code Review: {owner}/{repo}/{path}  (branch: {branch})\n"
        f"   {lines_count} lines  |  {len(code):,} characters\n"
        f"{'─' * 60}"
    )

    code_preview = _truncate(code, 1500)
    code_section = f"📄 Code Preview:\n```python\n{code_preview}\n```"

    exec_section_parts = []
    if exec_result["timed_out"]:
        exec_section_parts.append("⏱️  Execution: Timed out after 10 seconds.")
    else:
        exec_section_parts.append(f"▶️  Exit code: {exec_result['exit_code']}")
        if exec_result["stdout"]:
            exec_section_parts.append(f"Output:\n{_truncate(exec_result['stdout'], 800)}")
        if exec_result["stderr"]:
            exec_section_parts.append(f"Stderr:\n{_truncate(exec_result['stderr'], 600)}")
    exec_section = "Execution Results:\n" + "\n".join(exec_section_parts)

    return "\n\n".join([header, code_section, exec_section, review])


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Starting Developer Tools MCP server...")
    print("Tools registered:")
    print("  • github_tool        — GitHub API (search, read, issues)")
    print("  • code_executor      — Python sandbox execution")
    print("  • code_writer        — Code generation + static analysis")
    print("  • github_run_review  — Fetch → Execute → Review pipeline")
    if not GITHUB_TOKEN:
        print("\n⚠️  Warning: GITHUB_TOKEN not set. Rate limit: 60 requests/hour.")
        print("   Set GITHUB_TOKEN in your .env file for 5,000 requests/hour.")
    mcp.run(transport="stdio")