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
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Ensure it exists in your .env file.")

    model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

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
- Identify the task type: derivative, integral, solve_equation, simplify, or evaluate.
- Use standard math notation with * for multiplication and ^ for powers.
- For equations: return ONLY the left side (the part that equals 0).
- Output ONLY valid JSON.

JSON format examples:

For derivative:
{
  "task": "derivative",
  "expression": "x^2 - 3*x",
  "variable": "x"
}

For integral:
{
  "task": "integral",
  "expression": "x^2 + 2*x",
  "variable": "x"
}

For equation:
{
  "task": "solve_equation",
  "expression": "x^2 + 3*x + 2",
  "variables": ["x"]
}

For simplify:
{
  "task": "simplify",
  "expression": "(x+2)^2"
}

For evaluate:
{
  "task": "evaluate",
  "expression": "2^3 + 4*5"
}
"""


def extract_math_with_llm(query: str) -> dict:
    response = call_llm(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=query,
        temperature=0.5
    )

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        # Fallback: try to extract a JSON object
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
    
    # Clean the expression
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


@mcp.tool()
def solve_math(query: str) -> str:
    """
    Comprehensive math solver. Give it any math problem in natural language or expression form.
    Handles: derivatives, integrals, equation solving, simplification, evaluation.

    Args:
        query: The math problem (e.g., "5 + 5", "derivative of x^2", "solve x^2 - 4 = 0")

    Returns:
        str: The solution as readable text.
    """

    # Step 1: LLM extraction
    try:
        parsed = extract_math_with_llm(query)
    except Exception as e:
        return f"Failed to parse math query: {str(e)}"

    task = parsed.get("task")
    expression = parsed.get("expression", "")

    # Handle different task types
    try:
        if task == "derivative":
            variable = parsed.get("variable", "x")
            x = sp.symbols(variable)
            expr = sp.sympify(expression)
            result = sp.diff(expr, x)
            return f"Derivative of {expression} with respect to {variable} = {result}"

        elif task == "integral":
            variable = parsed.get("variable", "x")
            x = sp.symbols(variable)
            expr = sp.sympify(expression)
            result = sp.integrate(expr, x)
            return f"Integral of {expression} d{variable} = {result} + C"

        elif task == "solve_equation":
            variables = parsed.get("variables", ["x"])
            if len(variables) != 1:
                return "Only single-variable equations are supported."

            var = variables[0]
            validate_expression(expression, variables)

            # Try symbolic solution first
            try:
                x = sp.symbols(var)
                sym_expr = sp.sympify(expression)
                symbolic_solutions = sp.solve(sym_expr, x)
                if symbolic_solutions:
                    sols = ", ".join(str(s) for s in symbolic_solutions)
                    return f"Solutions for {expression} = 0: {var} = {sols}"
            except (sp.SympifyError, TypeError, ValueError):
                pass

            # Fallback to numerical solution
            func = build_numeric_function(expression, var)
            solutions = solve_numerically(func)
            sols = ", ".join(str(s) for s in solutions)
            return f"Numerical solutions for {expression} = 0: {var} ≈ {sols}"

        elif task == "simplify":
            expr = sp.sympify(expression)
            result = sp.simplify(expr)
            return f"Simplified: {expression} = {result}"

        elif task == "evaluate":
            expr = sp.sympify(expression)
            result = expr.evalf()
            return f"{expression} = {result}"

        else:
            return f"Unsupported task type: {task}. Supported: derivative, integral, solve_equation, simplify, evaluate"

    except Exception as e:
        return f"Failed to process {task}: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")