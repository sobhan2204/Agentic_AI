import mcp
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP
import sympy as sp
from sympy import sympify, N, simplify, solve, diff, integrate, limit, series
from sympy.parsing.sympy_parser import parse_expr
import math

mcp = FastMCP("maths-mcp-server")

@mcp.tool()
def calculate(
    expression: str,
    precision: int = 10
) -> Dict[str, Any]:
    """
    Perform mathematical calculations and solve expressions using SymPy.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2+2", "sin(pi/2)", "sqrt(16)", "x**2 + 2*x + 1")
        precision: Number of decimal places for the result
    
    Returns:
        Dict with calculation result and expression details
    """
    try:
        # Parse the expression using SymPy
        expr = parse_expr(expression)
        
        # Check if expression has variables
        variables = expr.free_symbols
        
        if variables:
            # If it has variables, return symbolic result
            simplified = simplify(expr)
            return {
                "expression": expression,
                "result": str(simplified),
                "variables": [str(var) for var in variables],
                "success": True,
                "type": "symbolic",
                "simplified": str(simplified)
            }
        else:
            # If no variables, evaluate numerically
            result = N(expr, precision)
            return {
                "expression": expression,
                "result": float(result) if result.is_real else str(result),
                "success": True,
                "type": "numerical"
            }
            
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
            "success": False
        }

@mcp.tool()
def solve_equation(
    equation: str,
    variable: str = "x"
) -> Dict[str, Any]:
    """
    Solve algebraic equations using SymPy.
    
    Args:
        equation: Equation to solve (e.g., "x**2 - 4", "2*x + 3 = 7")
        variable: Variable to solve for (default: "x")
    
    Returns:
        Dict with solutions
    """
    try:
        # Handle equations with "=" sign
        if "=" in equation:
            left, right = equation.split("=")
            expr = parse_expr(left) - parse_expr(right)
        else:
            expr = parse_expr(equation)
        
        var = sp.Symbol(variable)
        solutions = solve(expr, var)
        
        return {
            "equation": equation,
            "variable": variable,
            "solutions": [str(sol) for sol in solutions],
            "numerical_solutions": [float(sol.evalf()) if sol.is_real else str(sol.evalf()) for sol in solutions],
            "success": True
        }
        
    except Exception as e:
        return {
            "equation": equation,
            "error": str(e),
            "success": False
        }

@mcp.tool()
def calculus_operations(
    expression: str,
    operation: str,
    variable: str = "x",
    **kwargs
) -> Dict[str, Any]:
    """
    Perform calculus operations (differentiate, integrate, limits).
    
    Args:
        expression: Mathematical expression
        operation: "derivative", "integral", "limit"
        variable: Variable for operation (default: "x")
        **kwargs: Additional parameters (e.g., limit_point for limits)
    
    Returns:
        Dict with calculus result
    """
    try:
        expr = parse_expr(expression)
        var = sp.Symbol(variable)
        
        if operation == "derivative":
            result = diff(expr, var)
            return {
                "expression": expression,
                "operation": "derivative",
                "variable": variable,
                "result": str(result),
                "success": True
            }
            
        elif operation == "integral":
            result = integrate(expr, var)
            return {
                "expression": expression,
                "operation": "integral", 
                "variable": variable,
                "result": str(result),
                "success": True
            }
            
        elif operation == "limit":
            limit_point = kwargs.get("limit_point", 0)
            result = limit(expr, var, limit_point)
            return {
                "expression": expression,
                "operation": "limit",
                "variable": variable,
                "limit_point": limit_point,
                "result": str(result),
                "success": True
            }
            
    except Exception as e:
        return {
            "expression": expression,
            "operation": operation,
            "error": str(e),
            "success": False
        }

@mcp.tool()
def convert_units(
    value: float,
    from_unit: str,
    to_unit: str,
    unit_type: str = "length"
) -> Dict[str, Any]:
    """
    Convert between different units of measurement using precise calculations.
    
    Args:
        value: Numeric value to convert
        from_unit: Source unit (e.g., "km", "miles", "celsius", "kg", "lb")
        to_unit: Target unit (e.g., "miles", "km", "fahrenheit", "lb", "kg")
        unit_type: Type of unit ("length", "weight", "temperature", "volume", "area")
    
    Returns:
        Dict with conversion result
    """
    try:
        # Length conversions
        length_conversions = {
            "km_to_miles": sp.Rational(1000, 1609.344),
            "miles_to_km": sp.Rational(1609.344, 1000),
            "m_to_ft": sp.Rational(10000, 3048),
            "ft_to_m": sp.Rational(3048, 10000),
            "cm_to_inch": sp.Rational(1, 2.54),
            "inch_to_cm": sp.Rational(254, 100)
        }
        
        # Weight conversions
        weight_conversions = {
            "kg_to_lb": sp.Rational(22046226218, 10000000000),
            "lb_to_kg": sp.Rational(45359237, 100000000),
            "g_to_oz": sp.Rational(1000, 28349.523125),
            "oz_to_g": sp.Rational(28349523125, 1000000000)
        }
        
        # Temperature conversions (special handling needed)
        def celsius_to_fahrenheit(c):
            return sp.Rational(9, 5) * c + 32
        
        def fahrenheit_to_celsius(f):
            return sp.Rational(5, 9) * (f - 32)
        
        def celsius_to_kelvin(c):
            return c + sp.Rational(27315, 100)
        
        def kelvin_to_celsius(k):
            return k - sp.Rational(27315, 100)
        
        # Volume conversions
        volume_conversions = {
            "l_to_gal": sp.Rational(1000, 3785.411784),
            "gal_to_l": sp.Rational(3785411784, 1000000000),
            "ml_to_floz": sp.Rational(1000, 29.5735296875),
            "floz_to_ml": sp.Rational(295735296875, 10000000000)
        }
        
        # Combine all conversions
        all_conversions = {**length_conversions, **weight_conversions, **volume_conversions}
        
        conversion_key = f"{from_unit}_to_{to_unit}"
        
        # Handle temperature conversions separately
        temp_conversions = {
            "celsius_to_fahrenheit": celsius_to_fahrenheit,
            "fahrenheit_to_celsius": fahrenheit_to_celsius,
            "celsius_to_kelvin": celsius_to_kelvin,
            "kelvin_to_celsius": kelvin_to_celsius
        }
        
        if conversion_key in temp_conversions:
            result = temp_conversions[conversion_key](value)
            result_float = float(N(result, 10))
        elif conversion_key in all_conversions:
            factor = all_conversions[conversion_key]
            result = factor * value
            result_float = float(N(result, 10))
        else:
            return {
                "error": f"Conversion from {from_unit} to {to_unit} not supported",
                "supported_conversions": list(all_conversions.keys()) + list(temp_conversions.keys()),
                "success": False
            }
        
        return {
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": result_float,
            "converted_unit": to_unit,
            "unit_type": unit_type,
            "exact_result": str(result) if 'result' in locals() else str(result_float),
            "success": True
        }
        
    except Exception as e:
        return {
            "original_value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "error": str(e),
            "success": False
        }

@mcp.tool()
def matrix_operations(
    matrix_a: List[List[float]],
    operation: str,
    matrix_b: Optional[List[List[float]]] = None
) -> Dict[str, Any]:
    """
    Perform matrix operations using SymPy.
    
    Args:
        matrix_a: First matrix as list of lists
        operation: "determinant", "inverse", "transpose", "multiply", "add", "eigenvalues"
        matrix_b: Second matrix for operations like multiply/add
    
    Returns:
        Dict with matrix operation result
    """
    try:
        mat_a = sp.Matrix(matrix_a)
        
        if operation == "determinant":
            result = mat_a.det()
            return {
                "matrix": matrix_a,
                "operation": "determinant",
                "result": float(N(result)),
                "exact_result": str(result),
                "success": True
            }
            
        elif operation == "inverse":
            result = mat_a.inv()
            return {
                "matrix": matrix_a,
                "operation": "inverse",
                "result": [[float(N(cell)) for cell in row] for row in result.tolist()],
                "success": True
            }
            
        elif operation == "transpose":
            result = mat_a.T
            return {
                "matrix": matrix_a,
                "operation": "transpose",
                "result": [[float(N(cell)) for cell in row] for row in result.tolist()],
                "success": True
            }
            
        elif operation == "eigenvalues":
            eigenvals = mat_a.eigenvals()
            return {
                "matrix": matrix_a,
                "operation": "eigenvalues",
                "result": [str(val) for val in eigenvals.keys()],
                "numerical_result": [float(N(val)) if val.is_real else str(N(val)) for val in eigenvals.keys()],
                "success": True
            }
            
        elif operation in ["multiply", "add"] and matrix_b:
            mat_b = sp.Matrix(matrix_b)
            
            if operation == "multiply":
                result = mat_a * mat_b
            else:  # add
                result = mat_a + mat_b
                
            return {
                "matrix_a": matrix_a,
                "matrix_b": matrix_b,
                "operation": operation,
                "result": [[float(N(cell)) for cell in row] for row in result.tolist()],
                "success": True
            }
            
    except Exception as e:
        return {
            "matrix_a": matrix_a,
            "operation": operation,
            "error": str(e),
            "success": False
        }

if __name__ == "__main__":
    mcp.run(transport="stdio")  # useful if we want to run the server in the terminal locally in this
    #we will get the input and output in the teminal itself