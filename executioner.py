from langchain_core.messages import HumanMessage


def summarize_result(text: str, max_length: int = 400) -> str:
    """Truncate long tool/model outputs to keep token usage low."""
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [output truncated]"


async def execute_plan(agent, plan, execution_hint=None):
    """Execute a structured plan by delegating each step to the agent.

    Expects plan of the form:
    {
      "goal": "...",
      "steps": [
        {"id": 1, "tool": "search_web", "input": "..."}
      ]
    }
    """

    execution_results = []
    accumulated_context = ""
    # Use a consistent thread ID to maintain context across executions
    config = {"configurable": {"thread_id": "main_conversation"}}

    for i, step in enumerate(plan.get("steps", [])):
        tool_name = step.get("tool", "none")
        # Support both "action" (old schema) and "input" (new schema)
        action = step.get("action") or step.get("input") or ""

        
        if tool_name and tool_name != "none":
            if tool_name in ["send_email", "gmail"] and accumulated_context:
                prompt = f"Use the {tool_name} tool to: {action}\n\nEmail body should always contain \n{accumulated_context}\n\n Also provide confirmation after sending the mail"
            elif tool_name == "web_search" and accumulated_context:
                prompt = f"Use the {tool_name} tool to: {action}\n\nsearched result should always be elaborated if you are certain and if you find any relevace in {accumulated_context} then use that too"
            else:
             prompt = f"""
              You must call exactly ONE tool.Tool name: {tool_name} Task: {action}Rules:
              - Do NOT explain
              - Do NOT answer directly
              - ONLY call the tool"""
        else:    
            prompt = action # for example -> if the tool is "none" or conn=versational then just use action as prompt

        # Add context from previous steps so later steps can use earlier results
        if accumulated_context:
            prompt = f"Previous step results:\n{accumulated_context}\n\n{prompt}\n\nIMPORTANT: Use the information from previous steps ONLY IF relevant to this task."

        if execution_hint:
            prompt += f"\nIMPORTANT: {execution_hint}"

        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
        )

        result = response["messages"][-1].content
        summarized = summarize_result(result, max_length=400)
        execution_results.append(summarized)
        
        # Accumulate context for next steps (keep full result, not truncated)
        accumulated_context += f"Step {i+1} ({tool_name}): {result}\n\n"

    final_answer = "\n".join(execution_results)
    return execution_results, final_answer
