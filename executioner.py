from langchain_core.messages import HumanMessage

async def execute_plan(agent, plan, execution_hint=None):
    execution_results = []
    config = {"configurable": {"thread_id": "main_conversation"}}

    for step in plan["steps"]:
        prompt = f"Execute this step: {step['action']}"

        # 🔁 Add retry hint if available
        if execution_hint:
            prompt += f"\nIMPORTANT: {execution_hint}"

        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config
        )

        result = response["messages"][-1].content
        execution_results.append(result)

    final_answer = "\n".join(execution_results)
    return execution_results, final_answer
