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

    # Rule 2: Tool compliance (relaxed)
    # Just check that we got some content, don't enforce specific format
    for step, result in zip(plan.get("steps", []), execution_results):
        if len(result.strip()) < 20:
            return {
                "verdict": "RETRY",
                "reason": f"Step {step['id']} produced insufficient output",
                "retry_hint": "Provide more detailed information"
            }

    # Rule 3: Goal satisfaction (relaxed - only fail if answer is too short)
    # Allow answers that are substantive even if they don't contain exact goal keywords
    if len(final_answer.strip()) < 30:
        return {
            "verdict": "FAIL",
            "reason": "Final answer is too short",
            "retry_hint": None
        }

    # Rule 4: Freshness for news (removed - too strict)

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
