"""
Code Execution Tool for ADK Integration

This tool provides safe code execution capabilities for use with
the Agent Development Kit (ADK) in ParallaxMind.
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import subprocess
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import ast
import time

logger = logging.getLogger(__name__)

class CodeExecutionTool:
    """
    ADK-compatible code execution tool for data analysis and computation.
    
    This tool provides secure code execution with proper sandboxing
    and output capture for ADK agent consumption.
    """
    
    def __init__(self, timeout: int = 30, max_output_size: int = 10000):
        """Initialize the code execution tool."""
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.allowed_imports = {
            # Data analysis and visualization
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
            # Scientific computing
            'scipy', 'sklearn', 'scikit-learn',
            # Standard library
            'math', 'statistics', 'datetime', 'json', 'csv', 're',
            'collections', 'itertools', 'functools', 'operator',
            # Utilities
            'requests', 'urllib', 'base64', 'hashlib'
        }
        
        self.blocked_imports = {
            'os', 'sys', 'subprocess', 'shutil', 'glob',
            'socket', 'threading', 'multiprocessing',
            '__builtin__', '__builtins__', 'builtins',
            'exec', 'eval', 'compile', 'open'
        }
        
        logger.info("Code execution tool initialized with safety constraints")
    
    async def execute_python(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        libraries: Optional[List[str]] = None,
        return_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Execute Python code safely and return results.
        
        Args:
            code: Python code to execute
            context: Optional context variables to provide
            libraries: Optional list of required libraries
            return_format: Output format ("json", "text", "html")
            
        Returns:
            Dictionary containing execution results and metadata
        """
        execution_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        try:
            # Validate code safety
            safety_check = await self._validate_code_safety(code)
            if not safety_check["safe"]:
                return {
                    "execution_id": execution_id,
                    "success": False,
                    "error": f"Code safety violation: {safety_check['reason']}",
                    "output": "",
                    "execution_time": 0,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Check required libraries
            if libraries:
                missing_libs = await self._check_required_libraries(libraries)
                if missing_libs:
                    return {
                        "execution_id": execution_id,
                        "success": False,
                        "error": f"Missing required libraries: {', '.join(missing_libs)}",
                        "output": "",
                        "execution_time": 0,
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Execute code in controlled environment
            result = await self._execute_in_sandbox(code, context or {})
            
            execution_time = time.time() - start_time
            
            return {
                "execution_id": execution_id,
                "success": result["success"],
                "output": result["output"],
                "error": result.get("error", ""),
                "variables": result.get("variables", {}),
                "plots": result.get("plots", []),
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "code_stats": {
                    "lines": len(code.strip().split('\n')),
                    "characters": len(code),
                    "imports_used": result.get("imports_used", [])
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Code execution error: {e}")
            
            return {
                "execution_id": execution_id,
                "success": False,
                "error": str(e),
                "output": "",
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
    
    async def execute_data_analysis(
        self,
        data: Union[str, Dict, List],
        analysis_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute common data analysis tasks.
        
        Args:
            data: Data to analyze (CSV path, JSON data, or list)
            analysis_type: Type of analysis ("summary", "correlation", "visualization", "regression")
            parameters: Analysis parameters
            
        Returns:
            Analysis results
        """
        params = parameters or {}
        
        if analysis_type == "summary":
            code = self._generate_summary_code(data, params)
        elif analysis_type == "correlation":
            code = self._generate_correlation_code(data, params)
        elif analysis_type == "visualization":
            code = self._generate_visualization_code(data, params)
        elif analysis_type == "regression":
            code = self._generate_regression_code(data, params)
        else:
            return {
                "success": False,
                "error": f"Unknown analysis type: {analysis_type}",
                "output": ""
            }
        
        return await self.execute_python(
            code=code,
            context={"data": data, "params": params},
            libraries=["pandas", "numpy", "matplotlib", "seaborn"]
        )
    
    async def execute_math_computation(
        self,
        expression: str,
        variables: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Execute mathematical computations safely.
        
        Args:
            expression: Mathematical expression to evaluate
            variables: Optional variables for the expression
            
        Returns:
            Computation result
        """
        vars_dict = variables or {}
        
        # Create safe math code
        code = f"""
import math
import numpy as np
from fractions import Fraction

# Define variables
{chr(10).join(f"{k} = {v}" for k, v in vars_dict.items())}

# Calculate result
try:
    result = {expression}
    print(f"Expression: {expression}")
    print(f"Result: {{result}}")
    
    # Additional information
    if isinstance(result, (int, float)):
        print(f"Type: {{type(result).__name__}}")
        if isinstance(result, float):
            print(f"Decimal: {{result:.10f}}")
            print(f"Scientific: {{result:.3e}}")
    
    # Store result for return
    computation_result = result
except Exception as e:
    print(f"Computation error: {{e}}")
    computation_result = None
"""
        
        return await self.execute_python(
            code=code,
            context=vars_dict,
            libraries=["math", "numpy"]
        )
    
    async def _validate_code_safety(self, code: str) -> Dict[str, Any]:
        """Validate that code is safe to execute."""
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Check for dangerous operations
            for node in ast.walk(tree):
                # Check imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.blocked_imports:
                            return {
                                "safe": False,
                                "reason": f"Blocked import: {alias.name}"
                            }
                        elif alias.name not in self.allowed_imports:
                            logger.warning(f"Unknown import: {alias.name}")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.blocked_imports:
                        return {
                            "safe": False,
                            "reason": f"Blocked module import: {node.module}"
                        }
                
                # Check for dangerous function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['exec', 'eval', 'compile', '__import__']:
                            return {
                                "safe": False,
                                "reason": f"Blocked function call: {node.func.id}"
                            }
                
                # Check for file operations
                elif isinstance(node, ast.Attribute):
                    if node.attr in ['system', 'popen', 'remove', 'rmdir']:
                        return {
                            "safe": False,
                            "reason": f"Blocked system operation: {node.attr}"
                        }
            
            return {"safe": True, "reason": "Code passed safety checks"}
            
        except SyntaxError as e:
            return {
                "safe": False,
                "reason": f"Syntax error: {str(e)}"
            }
        except Exception as e:
            return {
                "safe": False,
                "reason": f"Code validation error: {str(e)}"
            }
    
    async def _check_required_libraries(self, libraries: List[str]) -> List[str]:
        """Check which required libraries are missing."""
        missing = []
        
        for lib in libraries:
            try:
                __import__(lib)
            except ImportError:
                missing.append(lib)
        
        return missing
    
    async def _execute_in_sandbox(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code in a controlled sandbox environment."""
        try:
            # Create execution environment
            exec_globals = {
                '__builtins__': {
                    # Safe built-ins only
                    'print': print,
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'sum': sum,
                    'min': min,
                    'max': max,
                    'abs': abs,
                    'round': round,
                    'sorted': sorted,
                    'reversed': reversed,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'type': type,
                    'isinstance': isinstance,
                    'hasattr': hasattr,
                    'getattr': getattr,
                    'setattr': setattr
                }
            }
            
            # Add context variables
            exec_globals.update(context)
            
            # Capture output
            import io
            import sys
            
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            exec_locals = {}
            plots = []
            imports_used = []
            
            try:
                # Execute the code
                exec(code, exec_globals, exec_locals)
                
                # Capture any matplotlib plots
                try:
                    import matplotlib.pyplot as plt
                    if plt.get_fignums():
                        # Save plots to temporary files
                        for fig_num in plt.get_fignums():
                            fig = plt.figure(fig_num)
                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                                fig.savefig(tmp.name, dpi=100, bbox_inches='tight')
                                plots.append({
                                    "figure_id": fig_num,
                                    "file_path": tmp.name,
                                    "format": "png"
                                })
                        plt.close('all')
                except ImportError:
                    pass
                
                # Get output
                stdout_content = stdout_capture.getvalue()
                stderr_content = stderr_capture.getvalue()
                
                # Limit output size
                if len(stdout_content) > self.max_output_size:
                    stdout_content = stdout_content[:self.max_output_size] + "\n... (output truncated)"
                
                # Extract variables (excluding built-ins and modules)
                result_variables = {}
                for name, value in exec_locals.items():
                    if not name.startswith('_') and not callable(value):
                        try:
                            # Try to serialize the value
                            json.dumps(value, default=str)
                            result_variables[name] = value
                        except (TypeError, ValueError):
                            result_variables[name] = str(value)
                
                return {
                    "success": True,
                    "output": stdout_content,
                    "error": stderr_content if stderr_content else None,
                    "variables": result_variables,
                    "plots": plots,
                    "imports_used": imports_used
                }
                
            except Exception as e:
                stderr_content = stderr_capture.getvalue()
                return {
                    "success": False,
                    "output": stdout_capture.getvalue(),
                    "error": stderr_content + str(e),
                    "variables": {},
                    "plots": [],
                    "imports_used": imports_used
                }
            
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "variables": {},
                "plots": [],
                "imports_used": []
            }
    
    def _generate_summary_code(self, data: Any, params: Dict) -> str:
        """Generate code for data summary analysis."""
        return """
import pandas as pd
import numpy as np

# Load or use provided data
if isinstance(data, str):
    # Assume it's a file path
    if data.endswith('.csv'):
        df = pd.read_csv(data)
    elif data.endswith('.json'):
        df = pd.read_json(data)
    else:
        print("Unsupported file format")
        df = None
elif isinstance(data, (list, dict)):
    df = pd.DataFrame(data)
else:
    df = data

if df is not None:
    print("Data Summary")
    print("=" * 40)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    print("Data Types:")
    print(df.dtypes)
    print()
    
    print("Statistical Summary:")
    print(df.describe())
    print()
    
    print("Missing Values:")
    print(df.isnull().sum())
    print()
    
    # Sample data
    print("Sample Data (first 5 rows):")
    print(df.head())
"""
    
    def _generate_correlation_code(self, data: Any, params: Dict) -> str:
        """Generate code for correlation analysis."""
        return """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data (same as summary)
if isinstance(data, str):
    if data.endswith('.csv'):
        df = pd.read_csv(data)
    elif data.endswith('.json'):
        df = pd.read_json(data)
    else:
        print("Unsupported file format")
        df = None
elif isinstance(data, (list, dict)):
    df = pd.DataFrame(data)
else:
    df = data

if df is not None:
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if not numeric_df.empty:
        print("Correlation Analysis")
        print("=" * 40)
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        print("Correlation Matrix:")
        print(corr_matrix)
        print()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix Heatmap')
        plt.tight_layout()
        plt.show()
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Strong correlation threshold
                    corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))
        
        if corr_pairs:
            print("Strong Correlations (|r| > 0.5):")
            for col1, col2, r in sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"{col1} <-> {col2}: {r:.3f}")
        else:
            print("No strong correlations found (|r| > 0.5)")
    else:
        print("No numeric columns found for correlation analysis")
"""
    
    def _generate_visualization_code(self, data: Any, params: Dict) -> str:
        """Generate code for data visualization."""
        chart_type = params.get('chart_type', 'histogram')
        
        return f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
if isinstance(data, str):
    if data.endswith('.csv'):
        df = pd.read_csv(data)
    elif data.endswith('.json'):
        df = pd.read_json(data)
    else:
        print("Unsupported file format")
        df = None
elif isinstance(data, (list, dict)):
    df = pd.DataFrame(data)
else:
    df = data

if df is not None:
    print("Data Visualization")
    print("=" * 40)
    
    # Set style
    plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
    
    # Create visualizations based on chart type
    chart_type = "{chart_type}"
    
    if chart_type == "histogram":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(min(len(numeric_cols), 4), 1, figsize=(10, 6*min(len(numeric_cols), 4)))
            if len(numeric_cols) == 1:
                axes = [axes]
            
            for i, col in enumerate(numeric_cols[:4]):
                if i < len(axes):
                    axes[i].hist(df[col].dropna(), bins=30, alpha=0.7)
                    axes[i].set_title(f'Distribution of {{col}}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.show()
    
    elif chart_type == "scatter":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6)
            plt.xlabel(numeric_cols[0])
            plt.ylabel(numeric_cols[1])
            plt.title(f'{{numeric_cols[0]}} vs {{numeric_cols[1]}}')
            plt.show()
    
    elif chart_type == "box":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            plt.figure(figsize=(12, 6))
            df[numeric_cols].boxplot()
            plt.title('Box Plot of Numeric Columns')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    print(f"Generated {{chart_type}} visualization")
"""
    
    def _generate_regression_code(self, data: Any, params: Dict) -> str:
        """Generate code for regression analysis."""
        return """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load data
if isinstance(data, str):
    if data.endswith('.csv'):
        df = pd.read_csv(data)
    elif data.endswith('.json'):
        df = pd.read_json(data)
    else:
        print("Unsupported file format")
        df = None
elif isinstance(data, (list, dict)):
    df = pd.DataFrame(data)
else:
    df = data

if df is not None:
    print("Regression Analysis")
    print("=" * 40)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) >= 2:
        # Use first column as target, rest as features
        target_col = numeric_cols[0]
        feature_cols = numeric_cols[1:]
        
        # Prepare data
        X = df[feature_cols].dropna()
        y = df[target_col].dropna()
        
        # Ensure X and y have same indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) > 1:
            # Fit regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            
            print(f"Target variable: {target_col}")
            print(f"Feature variables: {list(feature_cols)}")
            print(f"R-squared: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print()
            
            # Coefficients
            print("Coefficients:")
            for feature, coef in zip(feature_cols, model.coef_):
                print(f"  {feature}: {coef:.4f}")
            print(f"  Intercept: {model.intercept_:.4f}")
            print()
            
            # Plot actual vs predicted
            plt.figure(figsize=(10, 6))
            plt.scatter(y, y_pred, alpha=0.6)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            plt.xlabel(f'Actual {target_col}')
            plt.ylabel(f'Predicted {target_col}')
            plt.title(f'Actual vs Predicted Values (R² = {r2:.3f})')
            plt.tight_layout()
            plt.show()
        else:
            print("Insufficient data for regression analysis")
    else:
        print("Need at least 2 numeric columns for regression analysis")
"""

# Create global instance for ADK integration
code_exec_tool = CodeExecutionTool()