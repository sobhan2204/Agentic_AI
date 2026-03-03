from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime
import re


def summarize_result(text: str, max_length: int = 500) -> str:
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [truncated]"


def parse_email_input(raw_input: str) -> dict:
    """
    Parse the pipe-format email input produced by the intent router.
    Format: "recipient=X | subject=Y | body=Z"
    """
    result = {"recipient": "", "subject": "Hello", "body": ""}

    if "|" in raw_input and "recipient=" in raw_input:
        for part in raw_input.split("|"):
            part = part.strip()
            if part.startswith("recipient="):
                result["recipient"] = part[len("recipient="):].strip()
            elif part.startswith("subject="):
                result["subject"] = part[len("subject="):].strip() or "Hello"
            elif part.startswith("body="):
                result["body"] = part[len("body="):].strip()

    return result


def build_step_prompt(tool_name: str, action: str, previous_output: str | None) -> str:
    """
    Build a self-contained prompt for a single tool step.
    The LLM decides HOW to call the tool — we just tell it WHAT we need.
    Previous output is injected as context when relevant.
    """

    # For email: reconstruct with previous output as body if body is empty
    if tool_name == "send_email":
        parsed = parse_email_input(action)

        # If body is empty, fill it with previous step's output (e.g. search results)
        if not parsed["body"] and previous_output:
            parsed["body"] = previous_output

        return (
            f"Send an email using the send_email tool with these exact parameters:\n"
            f"  recipient: {parsed['recipient']}\n"
            f"  subject:   {parsed['subject']}\n"
            f"  body:      {parsed['body'] or 'Hello!'}\n\n"
            f"Call the send_email tool now. Do not explain or summarize — just call it."
        )

    # For all other tools: give the LLM the task + context and let it call the tool
    context_block = ""
    if previous_output:
        context_block = (
            f"\nContext from previous step (use this as input if needed):\n"
            f"{previous_output}\n"
        )

    return (
        f"Call the {tool_name} tool to complete this task:\n"
        f"Task: {action}"
        f"{context_block}\n"
        f"Rules: Call the tool directly. Do NOT answer from memory. Do NOT explain."
    )


async def execute_plan(agent, plan, execution_hint=None, config=None):
    """
    Execute a plan as a strict pipeline: output of step N → input of step N+1.

    Key design decisions:
    - Each step runs in its OWN fresh thread so MemorySaver history never
      bleeds between steps and confuses the agent.
    - The previous step's output is passed EXPLICITLY in the prompt —
      not through shared memory.
    - The LLM decides how to call the tool — we just tell it what we need.
    - Synthesis runs last in its own thread for the user-facing reply.
    """
    if config is None:
        config = {"configurable": {"thread_id": "main"}}

    base_thread = config["configurable"]["thread_id"]

    steps             = plan.get("steps", [])
    execution_results = []
    previous_output   = None   # output of last step, piped into next step

    for i, step in enumerate(steps):
        tool_name = step.get("tool", "none")
        action    = step.get("action") or step.get("input") or ""

        # Each step gets a fresh thread — no history bleed between steps
        step_config = {"configurable": {"thread_id": f"{base_thread}_step{i}"}}

        prompt = build_step_prompt(tool_name, action, previous_output)

        if execution_hint:
            prompt += f"\nIMPORTANT: {execution_hint}"

        try:
            response = await agent.ainvoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=step_config,
            )
            raw_result = response["messages"][-1].content
        except Exception as e:
            raw_result = f"Step {i+1} ({tool_name}) failed: {str(e)[:200]}"

        summarized = summarize_result(raw_result, max_length=500)
        execution_results.append(summarized)

        # Pass this step's output to the next step
        previous_output = summarized

    # ── Synthesis: user-facing reply only ─────────────────────────────────
    # Runs in its own thread. Purely for what the assistant says in chat.
    # This is completely separate from what the tools already executed.
    synthesis_config = {"configurable": {"thread_id": f"{base_thread}_synthesis"}}

    context = "\n\n".join(
        f"Step {i+1} ({plan['steps'][i]['tool']}): {r}"
        for i, r in enumerate(execution_results)
    )

    if context.strip():
        synthesis_prompt = (
            f"Based on these tool results, write a clear friendly reply for the user.\n"
            f"Do NOT call any tools. Just summarize what was done and what was found.\n\n"
            f"Results:\n{context}\n\n"
            f"Goal: {plan.get('goal', '')}\n\n"
            f"Reply:"
        )
        try:
            synthesis_response = await agent.ainvoke(
                {"messages": [HumanMessage(content=synthesis_prompt)]},
                config=synthesis_config,
            )
            final_answer = synthesis_response["messages"][-1].content
        except Exception:
            final_answer = "\n".join(execution_results)
    else:
        final_answer = "\n".join(execution_results)

    return execution_results, final_answer