"""
Calculator Assistant

A focused assistant for mathematical calculations and numeric operations.
Demonstrates the assistant pattern with computational capabilities.
"""

import time
import math
from typing import Any, Dict, Union
from swarm_system.assistants.base_assistant import BaseAssistant, AssistantConfig


class CalculatorAssistant(BaseAssistant):
    """
    Assistant specialized in mathematical calculations and numeric operations.

    Capabilities:
    - Basic arithmetic operations
    - Advanced mathematical functions
    - Statistical calculations
    - Unit conversions
    """

    def get_default_prompt(self) -> str:
        return """You are a Calculator Assistant specialized in performing mathematical calculations and numeric analysis.

Your capabilities include:
- Basic arithmetic (addition, subtraction, multiplication, division)
- Advanced functions (square root, power, logarithm, trigonometry)
- Statistical operations (mean, median, standard deviation)
- Unit conversions (temperature, length, weight, etc.)

Always provide clear, structured responses with step-by-step calculations and explanations."""

    async def execute_async(self, input_data: Any, **kwargs) -> Any:
        """
        Execute mathematical calculations.

        Args:
            input_data: Mathematical expression or calculation specification
            **kwargs: Additional parameters (operation, precision, etc.)

        Returns:
            Calculation results with step-by-step breakdown
        """
        start_time = time.time()

        try:
            # Handle different input formats
            if isinstance(input_data, str):
                expression = input_data
                operation = kwargs.get('operation', 'evaluate')
            elif isinstance(input_data, dict):
                expression = input_data.get('expression', '')
                operation = input_data.get('operation', 'evaluate')
            else:
                expression = str(input_data)
                operation = kwargs.get('operation', 'evaluate')

            # Perform the requested operation
            if operation == 'evaluate':
                result = self._evaluate_expression(expression)
            elif operation == 'statistics':
                data = kwargs.get('data', [])
                result = self._calculate_statistics(data)
            elif operation == 'convert':
                value = float(kwargs.get('value', 0))
                from_unit = kwargs.get('from_unit', '')
                to_unit = kwargs.get('to_unit', '')
                result = self._convert_units(value, from_unit, to_unit)
            else:
                result = self._evaluate_expression(expression)  # Default to evaluation

            execution_time = time.time() - start_time
            self._record_execution(input_data, result, execution_time)

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_result = {
                "error": str(e),
                "operation": operation,
                "execution_time": execution_time
            }
            self._record_execution(input_data, error_result, execution_time)
            return error_result

    def _evaluate_expression(self, expression: str) -> Dict[str, Any]:
        """Evaluate a mathematical expression safely."""
        try:
            # Basic evaluation with limited functions for security
            allowed_names = {
                'sqrt': math.sqrt, 'pow': pow, 'abs': abs,
                'round': round, 'ceil': math.ceil, 'floor': math.floor,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'pi': math.pi, 'e': math.e
            }

            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, allowed_names)

            return {
                "operation": "evaluate",
                "expression": expression,
                "result": result,
                "type": type(result).__name__
            }

        except Exception as e:
            return {
                "operation": "evaluate",
                "expression": expression,
                "error": str(e),
                "result": None
            }

    def _calculate_statistics(self, data: list) -> Dict[str, Any]:
        """Calculate basic statistics for a dataset."""
        if not data:
            return {"error": "No data provided for statistical analysis"}

        try:
            data = [float(x) for x in data]
            n = len(data)

            mean = sum(data) / n
            sorted_data = sorted(data)
            median = sorted_data[n//2] if n % 2 == 1 else (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2

            variance = sum((x - mean) ** 2 for x in data) / n
            std_dev = math.sqrt(variance)

            return {
                "operation": "statistics",
                "count": n,
                "mean": mean,
                "median": median,
                "standard_deviation": std_dev,
                "min": min(data),
                "max": max(data),
                "range": max(data) - min(data)
            }

        except Exception as e:
            return {
                "operation": "statistics",
                "error": str(e)
            }

    def _convert_units(self, value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        """Convert between different units."""
        # Simple conversion factors (can be expanded)
        conversions = {
            # Temperature
            ('celsius', 'fahrenheit'): lambda x: (x * 9/5) + 32,
            ('fahrenheit', 'celsius'): lambda x: (x - 32) * 5/9,
            # Length
            ('meters', 'feet'): lambda x: x * 3.28084,
            ('feet', 'meters'): lambda x: x / 3.28084,
            ('kilometers', 'miles'): lambda x: x * 0.621371,
            ('miles', 'kilometers'): lambda x: x / 0.621371,
        }

        conversion_key = (from_unit.lower(), to_unit.lower())

        if conversion_key in conversions:
            try:
                converted_value = conversions[conversion_key](value)
                return {
                    "operation": "convert",
                    "from_value": value,
                    "from_unit": from_unit,
                    "to_value": converted_value,
                    "to_unit": to_unit,
                    "conversion_factor": converted_value / value if value != 0 else 0
                }
            except Exception as e:
                return {
                    "operation": "convert",
                    "error": str(e)
                }
        else:
            return {
                "operation": "convert",
                "error": f"Conversion from {from_unit} to {to_unit} not supported"
            }


def create_calculator_assistant(name: str = "calculator") -> CalculatorAssistant:
    """Factory function to create a calculator assistant."""
    config = AssistantConfig(
        name=name,
        description="Specialized assistant for mathematical calculations and numeric operations",
        model_id="llama3.2",  # Lightweight model for calculations
        system_prompt=None,  # Will use get_default_prompt()
        tools=[],  # No external tools needed for basic calculations
        metadata={
            "version": "1.0.0",
            "capabilities": ["evaluate", "statistics", "convert"],
            "model_size": "lightweight"
        }
    )

    return CalculatorAssistant(config)
