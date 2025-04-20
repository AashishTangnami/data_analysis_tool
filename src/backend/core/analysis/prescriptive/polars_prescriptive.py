
import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional
from core.analysis.prescriptive.base import PrescriptiveAnalysisBase

class PolarsPrescriptiveAnalysis(PrescriptiveAnalysisBase):
    """
    Polars implementation of prescriptive analysis strategy.
    """
    
    def analyze(self, data: pl.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform prescriptive analysis using polars.
        
        Args:
            data: polars DataFrame
            params: Parameters for the analysis
                - objective_column: Column to optimize
                - objective_type: 'maximize' or 'minimize'
                - decision_variables: List of variables to adjust
                - constraints: List of constraints on decision variables
            
        Returns:
            Dictionary containing analysis results
        """
        # For optimization, we'll convert to pandas and use scipy
        import pandas as pd
        from scipy.optimize import minimize, LinearConstraint
        from sklearn.linear_model import LinearRegression
        
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
        selected_data = selected_data.drop_nulls()
        
        # Convert to pandas for optimization
        pandas_df = selected_data.to_pandas()
        
        # Build a simple predictive model for the objective
        X = pandas_df[decision_variables]
        y = pandas_df[objective_column]
        
        # Convert categorical variables to numeric
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = X[col].astype('category').cat.codes
        
        # Train a linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Define the objective function for optimization
        def objective_function(x):
            # Reshape to match sklearn's expected format
            x_reshaped = x.reshape(1, -1)
            # Predict the objective value
            prediction = model.predict(x_reshaped)[0]
            # Flip sign if maximizing
            return -prediction if objective_type == "maximize" else prediction
        
        # Set up constraints
        bounds = []
        for var in decision_variables:
            # Find constraint for this variable
            var_constraint = next((c for c in constraints if c.get("variable") == var), None)
            
            if var_constraint:
                min_val = var_constraint.get("min", 0)
                max_val = var_constraint.get("max", 100)
                bounds.append((min_val, max_val))
            else:
                # Default bounds
                bounds.append((0, 100))
        
        # Set up initial point (mean of each decision variable)
        x0 = X.mean().values
        
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
            x_opt = optimization_result.x.reshape(1, -1)
            objective_value = model.predict(x_opt)[0]
            
            results["optimization_results"] = {
                "optimal_values": optimal_values,
                "objective_value": float(objective_value),
                "convergence": bool(optimization_result.success),
                "message": str(optimization_result.message)
            }
            
            # Generate scenario comparison
            current_scenario = {
                "scenario": "Current Average",
                "values": {var: float(X[var].mean()) for var in decision_variables},
                "objective_value": float(y.mean())
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
                for j, var in enumerate(decision_variables):
                    # Random value within bounds
                    lower, upper = bounds[j]
                    random_values[var] = float(np.random.uniform(lower, upper))
                
                # Predict the objective value
                x_random = np.array([random_values[var] for var in decision_variables]).reshape(1, -1)
                objective_value = model.predict(x_random)[0]
                
                alternative_scenarios.append({
                    "scenario": f"Alternative {i+1}",
                    "values": random_values,
                    "objective_value": float(objective_value)
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
                obj_plus = model.predict(x_plus.reshape(1, -1))[0]
                obj_minus = model.predict(x_minus.reshape(1, -1))[0]
                obj_optimal = objective_value
                
                # Calculate sensitivity
                sensitivity[var] = {
                    "increase_effect": float(obj_plus - obj_optimal),
                    "decrease_effect": float(obj_minus - obj_optimal),
                    "sensitivity_score": float(abs(obj_plus - obj_minus) / (2 * delta))
                }
            
            results["sensitivity_analysis"] = sensitivity
        
        except Exception as e:
            results["optimization_results"] = {"error": str(e)}
            results["scenario_comparison"] = []
            results["sensitivity_analysis"] = {}
        
        return results