import re

STOPWORDS = {"of", "the", "a", "an", "to", "for", "and"}

def rule_based_verifier(user_query, plan, execution_results, final_answer):

    plan_steps = len(plan.get("steps", []))
    execution_steps = len(execution_results)

    # Rule 1: Step count validation
    if execution_steps < plan_steps:
        return {
            "verdict": "RETRY",
            "reason": "Not all planned steps were executed",
            "retry_hint": "Execute all remaining steps in the plan"
        }

    if execution_steps > plan_steps:
        return {
            "verdict": "FAIL",
            "reason": "Executor produced extra unexpected steps",
            "retry_hint": None
        }

    # Rule 2: Tool compliance
    for step, result in zip(plan.get("steps", []), execution_results):
        tool = step.get("tool")

        if tool == "websearch":
            if not any(
                k in result.lower()
                for k in ["http", "www", "news", "source", "reported", "according"]
            ):
                return {
                    "verdict": "RETRY",
                    "reason": f"Step {step['id']} expected web search output",
                    "retry_hint": "Redo web search and include sources"
                }

    # Rule 3: Goal satisfaction (soft check)
    goal_keywords = [
        w for w in plan["goal"].lower().split()
        if w not in STOPWORDS and len(w) > 3
    ]

    if goal_keywords and not any(
        word in final_answer.lower() for word in goal_keywords
    ):
        return {
            "verdict": "FAIL",
            "reason": "Final answer does not satisfy the planned goal",
            "retry_hint": None
        }

    # Rule 4: Freshness for news
    if any(k in user_query.lower() for k in ["news", "latest", "current"]):
        if not re.search(r"\b(20\d{2}|today|yesterday|this week|this month)\b",
                         final_answer.lower()):
            return {
                "verdict": "RETRY",
                "reason": "News output lacks freshness indicators",
                "retry_hint": "Include recent date and source"
            }

    # Rule 5: Generic failure text
    generic_failures = [
        "i am not sure",
        "as an ai",
        "i cannot find",
        "no information available"
    ]

    if any(p in final_answer.lower() for p in generic_failures):
        return {
            "verdict": "FAIL",
            "reason": "Final answer is generic or evasive",
            "retry_hint": None
        }


    return {
        "verdict": "PASS",
        "reason": "All verification rules passed",
        "retry_hint": None
    }
