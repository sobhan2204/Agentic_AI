import json
import os
import numpy as np
import sympy as sp
from scipy.optimize import fsolve
from mcp.server.fastmcp import FastMCP
import requests
from dotenv import load_dotenv

mcp = FastMCP("math_server")

load_dotenv()

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0) -> str:
    """
    Calls Groq's OpenAI-compatible Chat Completions API and returns the content string.

    Env vars:
      - GROQ_API_KEY: required
      - GROQ_MODEL: optional (default: "llama-3.3-70b-versatile")
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Ensure it exists in your .env file.")

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature) if temperature is not None else 0.0,
        "max_tokens": 800,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        # Surface helpful error details but avoid leaking secrets
        try:
            details = resp.json()
        except Exception:
            details = {"error": resp.text[:500]}
        raise RuntimeError(f"Groq API error {resp.status_code}: {details}")

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("Unexpected Groq API response format")

    return content

EXTRACTION_SYSTEM_PROMPT = """
You are a mathematical query parser.

Rules:
- Do NOT solve the problem.
- Convert natural language math into structured JSON.
- Use standard math notation.
- For the expression: return ONLY the left side of the equation (the part that equals 0).
  Example: For "x^2 + 3x + 2 = 0", return expression as "x^2 + 3*x + 2" (NOT "x^2 + 3*x + 2 = 0").
- Use ^ for powers and * for multiplication in the expression.
- Output ONLY valid JSON.

JSON format:
{
  "task": "solve_equation",
  "expression": "<expression>",
  "variables": ["x"]
}
"""


def extract_math_with_llm(query: str) -> dict:
    response = call_llm(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=query,
        temperature=0
    )

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        # Fallback: try to extract a JSON object if wrapped in fences or extra text
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = response[start:end+1]
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                raise ValueError("LLM did not return valid JSON")
        else:
            raise ValueError("LLM did not return valid JSON")
    
    # Clean the expression: remove trailing "= 0" if present, and convert ^ to **
    if "expression" in parsed:
        expr = parsed["expression"].strip()
        # Remove "= 0" at the end if present
        if expr.endswith("= 0"):
            expr = expr[:-3].strip()
        elif expr.endswith("=0"):
            expr = expr[:-2].strip()
        # Convert ^ to ** for Python compatibility
        expr = expr.replace("^", "**")
        parsed["expression"] = expr
    
    return parsed



def validate_expression(expr: str, variables: list[str]) -> None:
    symbols = {v: sp.symbols(v) for v in variables}
    sp.sympify(expr, locals=symbols)



def build_numeric_function(expr: str, var: str):
    x = sp.symbols(var)
    sym_expr = sp.sympify(expr)
    func = sp.lambdify(x, sym_expr, modules=["numpy"])
    return func

def solve_numerically(func, initial_guess=1.0):
    sol = fsolve(func, x0=initial_guess)
    return sol.tolist()


@mcp.tool()#(name="solve_math", description="Solve mathematical equations from natural language queries")
def solve_math(query: str) -> dict:
    """
    Agentic math solver:
    - LLM extracts equation
    - SciPy solves numerically
    """

    # Step 1: LLM extraction
    parsed = extract_math_with_llm(query)

    if parsed.get("task") != "solve_equation":
        return {
            "error": "Only equation solving is supported in this tool."
        }

    expression = parsed["expression"]
    variables = parsed["variables"]

    if len(variables) != 1:
        return {
            "error": "Only single-variable equations are supported."
        }

    var = variables[0]

    # Step 2: Validate
    try:
        validate_expression(expression, variables)
    except Exception as e:
        return {
            "error": f"Invalid mathematical expression: {str(e)}"
        }

    # Step 3: Build numeric function
    try:
        func = build_numeric_function(expression, var)
    except Exception as e:
        return {
            "error": f"Failed to build numeric function: {str(e)}"
        }

    # Step 4: Solve
    try:
        solutions = solve_numerically(func)
    except Exception as e:
        return {
            "error": f"Numerical solver failed: {str(e)}"
        }

    return {
        "query": query,
        "expression": f"{expression} = 0",
        "variable": var,
        "solutions": solutions,
        "method": "numerical (SciPy fsolve)"
    }


# if __name__ == "__main__":
#     # For local testing
#     test_queries = [
#         "Solve for x: 2x^2 - 4x - 6 = 0",
#         "Find the roots of the equation x^3 - 6x^2 + 11x - 6 = 0",
#         "What is x if x^2 + 3x + 2 = 0?"
#     ]

#     for query in test_queries:
#         result = solve_math(query)
#         print(f"Query: {query}")
#         print(f"Result: {result}")
#         print("-" * 40)

if __name__ == "__main__":
    mcp.run(transport="stdio")