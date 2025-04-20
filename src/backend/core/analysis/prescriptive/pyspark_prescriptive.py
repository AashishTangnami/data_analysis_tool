
from typing import Dict, Any
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from core.analysis.prescriptive.base import PrescriptiveAnalysisBase

class PySparkPrescriptiveAnalysis(PrescriptiveAnalysisBase):
    """
    PySpark implementation of prescriptive analysis strategy.
    """
    
    def analyze(self, data: SparkDataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform prescriptive analysis using PySpark.
        
        Args:
            data: PySpark DataFrame
            params: Parameters for the analysis
                - objective_column: Column to optimize
                - objective_type: 'maximize' or 'minimize'
                - decision_variables: List of variables to adjust
                - constraints: List of constraints on decision variables
            
        Returns:
            Dictionary containing analysis results
        """
        # For optimization, we'll partially convert to pandas and use scipy
        import pandas as pd
        import numpy as np
        from scipy.optimize import minimize
        
        # Extract parameters
        objective_column = params.get("objective_column")
        objective_type = params.get("objective_type", "maximize")
        decision_variables = params.get("decision_variables", [])
        constraints = params.get("constraints", [])
        
        # Validate inputs
        if not objective_column or objective_column not in data.columns:
            raise ValueError(f"Objective column '{objective_column}' not found in data")
        
        decision_variables = [col for col in decision_variables if col in data.columns]
        if not decision_variables:
            raise ValueError("No valid decision variables provided")
        
        # Initialize results
        results = {
            "optimization_results": {},
            "scenario_comparison": [],
            "sensitivity_analysis": {}
        }
        
        # Select relevant columns
        selected_data = data.select([objective_column] + decision_variables)
        
        # Handle missing values for analysis
        selected_data = selected_data.dropna()
        
        # Create feature vector for model training
        assembler = VectorAssembler(inputCols=decision_variables, outputCol="features")
        vector_data = assembler.transform(selected_data)
        
        # Build a linear regression model for the objective
        lr = LinearRegression(featuresCol="features", labelCol=objective_column)
        model = lr.fit(vector_data)
        
        # Extract coefficients and intercept for optimization
        coefficients = model.coefficients.toArray()
        intercept = model.intercept
        
        # Get min/max values and means for decision variables
        stats = {}
        for var in decision_variables:
            min_val = selected_data.agg(F.min(var)).collect()[0][0]
            max_val = selected_data.agg(F.max(var)).collect()[0][0]
            mean_val = selected_data.agg(F.mean(var)).collect()[0][0]
            stats[var] = {"min": min_val, "max": max_val, "mean": mean_val}
        
        # Define the objective function for optimization
        def objective_function(x):
            # Calculate prediction using the linear model's coefficients
            prediction = np.dot(x, coefficients) + intercept
            # Flip sign if maximizing
            return -prediction if objective_type == "maximize" else prediction
        
        # Set up constraints
        bounds = []
        for i, var in enumerate(decision_variables):
            # Find constraint for this variable
            var_constraint = next((c for c in constraints if c.get("variable") == var), None)
            
            if var_constraint:
                min_val = var_constraint.get("min", stats[var]["min"])
                max_val = var_constraint.get("max", stats[var]["max"])
                bounds.append((min_val, max_val))
            else:
                # Default bounds
                bounds.append((stats[var]["min"], stats[var]["max"]))
        
        # Set up initial point (mean of each decision variable)
        x0 = np.array([stats[var]["mean"] for var in decision_variables])
        
        # Run optimization
        try:
            optimization_result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds
            )
            
            # Process optimization results
            optimal_values = {}
            for i, var in enumerate(decision_variables):
                optimal_values[var] = float(optimization_result.x[i])
            
            # Calculate the objective value
            objective_value = np.dot(optimization_result.x, coefficients) + intercept
            if objective_type == "maximize":
                objective_value = -objective_function(optimization_result.x)
            
            results["optimization_results"] = {
                "optimal_values": optimal_values,
                "objective_value": float(objective_value),
                "convergence": bool(optimization_result.success),
                "message": str(optimization_result.message)
            }
            
            # Generate scenario comparison
            current_scenario = {
                "scenario": "Current Average",
                "values": {var: float(stats[var]["mean"]) for var in decision_variables},
                "objective_value": float(np.dot([stats[var]["mean"] for var in decision_variables], coefficients) + intercept)
            }
            
            optimal_scenario = {
                "scenario": "Optimal",
                "values": optimal_values,
                "objective_value": float(objective_value)
            }
            
            # Create a few more scenarios for comparison
            alternative_scenarios = []
            for i in range(3):
                # Generate a random perturbation from the optimal
                random_values = {}
                random_array = []
                for j, var in enumerate(decision_variables):
                    # Random value within bounds
                    lower, upper = bounds[j]
                    random_val = float(np.random.uniform(lower, upper))
                    random_values[var] = random_val
                    random_array.append(random_val)
                
                # Predict the objective value
                random_objective = np.dot(random_array, coefficients) + intercept
                
                alternative_scenarios.append({
                    "scenario": f"Alternative {i+1}",
                    "values": random_values,
                    "objective_value": float(random_objective)
                })
            
            results["scenario_comparison"] = [current_scenario, optimal_scenario] + alternative_scenarios
            
            # Perform sensitivity analysis
            sensitivity = {}
            for i, var in enumerate(decision_variables):
                # Change variable by +/- 10% and observe objective change
                delta = 0.1 * optimal_values[var]
                
                # Copy optimal values
                x_plus = optimization_result.x.copy()
                x_minus = optimization_result.x.copy()
                
                # Apply changes within bounds
                x_plus[i] = min(bounds[i][1], x_plus[i] + delta)
                x_minus[i] = max(bounds[i][0], x_minus[i] - delta)
                
                # Calculate new objective values
                obj_plus = np.dot(x_plus, coefficients) + intercept
                obj_minus = np.dot(x_minus, coefficients) + intercept
                
                # Calculate sensitivity
                sensitivity[var] = {
                    "increase_effect": float(obj_plus - objective_value),
                    "decrease_effect": float(obj_minus - objective_value),
                    "sensitivity_score": float(abs(obj_plus - obj_minus) / (2 * delta))
                }
            
            results["sensitivity_analysis"] = sensitivity
        
        except Exception as e:
            results["optimization_results"] = {"error": str(e)}
            results["scenario_comparison"] = []
            results["sensitivity_analysis"] = {}
        
        return results