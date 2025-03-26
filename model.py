import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize


class WeightedEnsembleGB:
    """Weighted ensemble model using Lasso, Random Forest, and Gradient Boosting for regression.
    - Trains three base learners (LassoCV, RandomForestRegressor, GradientBoostingRegressor)
    - Optimizes hyperparameters using GridSearchCV
    - Uses validation data to optimize the ensemble weights
    - Uses cross-validation to assess performance

    Attributes:
    lasso_pipeline (Pipeline): Lasso regression pipeline with imputation and scaling
    rf_pipeline (Pipeline): Random Forest pipeline with imputation
    gb_pipeline (Pipeline): Gradient Boosting pipeline with imputation
    lasso_weight (float): Weight assigned to Lasso predictions
    rf_weight (float): Weight assigned to Random Forest predictions
    gb_weight (float): Weight assigned to Gradient Boosting predictions
    """
    def __init__(self, lasso_params=None, rf_params=None, gb_params=None):
        self.lasso_pipeline = Pipeline([
            ('imputer', IterativeImputer(max_iter=10, random_state=42)),
            ('scaler', StandardScaler()),
            ('model', LassoCV(cv=5, random_state=42))
        ])

        if rf_params is None: # default params
            rf_params = {
                'n_estimators': 100,
                'max_depth': 12,
                'min_samples_split': 5,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        
        self.rf_pipeline = Pipeline([
            ('imputer', IterativeImputer(max_iter=10, random_state=42)),
            ('model', RandomForestRegressor(**rf_params))
        ])

        if gb_params is None: # default params
            gb_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            }
        
        self.gb_pipeline = Pipeline([
            ('imputer', IterativeImputer(max_iter=10, random_state=42)),
            ('model', GradientBoostingRegressor(**gb_params))
        ])
        
        # initialize weights to optimize during fit
        self.lasso_weight = 0.33
        self.rf_weight = 0.33
        self.gb_weight = 0.34
    
    def tune_hyperparameters(self, X, y):
        """Tune hyperparameters using GridSearchCV for Lasso, RandomForest, and GradientBoosting."""
        print("Tuning Lasso hyperparameters...")
        self.lasso_pipeline.named_steps['model'] = LassoCV(
            alphas=np.logspace(-4, 1, 10),
            cv=5,
            random_state=42
        )
        self.lasso_pipeline.fit(X, y)
        best_lasso_alpha = self.lasso_pipeline.named_steps['model'].alpha_
        print(f"Best Lasso alpha: {best_lasso_alpha}")

        print("Tuning RandomForest hyperparameters...")
        rf_grid = {
            'model__n_estimators': [50, 100, 150],
            'model__max_depth': [10, 12, 15],
            'model__min_samples_split': [2, 5, 10]
        }
        self.rf_pipeline = GridSearchCV(self.rf_pipeline, rf_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
        self.rf_pipeline.fit(X, y)
        best_rf_params = self.rf_pipeline.best_params_
        print(f"Best RF params: {best_rf_params}")
    
        print("Tuning GradientBoostingRegressor hyperparameters...")
        gb_grid = {
            'model__n_estimators': [50, 100, 150],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 5, 7]
        }
        self.gb_pipeline = GridSearchCV(self.gb_pipeline, gb_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
        self.gb_pipeline.fit(X, y)
        best_gb_params = self.gb_pipeline.best_estimator_.named_steps['model'].get_params()
        print(f"Best GB params: {best_gb_params}")

    def fit(self, X_train, y_train, X_val=None, y_val=None, cv_folds=5, tune=False):
        """Train the model with weight optimization and tuning."""
        if tune: # optional for faster running
            self.tune_hyperparameters(X_train, y_train)
        
        self.lasso_pipeline.fit(X_train, y_train)
        self.rf_pipeline.fit(X_train, y_train)
        self.gb_pipeline.fit(X_train, y_train)

        # optimize model weights to minimize validation MSE
        if X_val is not None and y_val is not None:
            lasso_pred_val = self.lasso_pipeline.predict(X_val)
            rf_pred_val = self.rf_pipeline.predict(X_val)
            gb_pred_val = self.gb_pipeline.predict(X_val)
            
            def objective(weights):
                weights = weights / np.sum(weights) # normalize weights
                
                ensemble_pred = (
                    weights[0] * lasso_pred_val + 
                    weights[1] * rf_pred_val + 
                    weights[2] * gb_pred_val
                )
                
                return mean_squared_error(y_val, ensemble_pred)
            
            initial_weights = np.array([0.33, 0.33, 0.34])
            
            # weights must be positive and sum to 1
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            bounds = [(0, 1), (0, 1), (0, 1)]
            
            result = minimize(
                objective, 
                initial_weights, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            optimized_weights = result.x
            self.lasso_weight = optimized_weights[0]
            self.rf_weight = optimized_weights[1]
            self.gb_weight = optimized_weights[2]
            
            print(f"Optimized weights: Lasso={self.lasso_weight:.3f}, RF={self.rf_weight:.3f}, GB={self.gb_weight:.3f}")
            
        return self

    def predict(self, X):
        """Make predictions using the weighted ensemble."""
        lasso_pred = self.lasso_pipeline.predict(X)
        rf_pred = self.rf_pipeline.predict(X)
        gb_pred = self.gb_pipeline.predict(X)
        
        # finalize predictions with optimized weights
        ensemble_pred = (
            self.lasso_weight * lasso_pred + 
            self.rf_weight * rf_pred + 
            self.gb_weight * gb_pred
        )
        
        return ensemble_pred

    def cross_validate_ensemble(self, X, y, cv_folds=5):
        """Perform cross-validation to assess ensemble performance and optimize weights."""
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        rmse_scores = []
    
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
            self.fit(X_train, y_train, X_val, y_val)
    
            y_pred = self.predict(X_val)
    
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmse_scores.append(rmse)
    
        mean_rmse = np.mean(rmse_scores)
        print(f"Cross-Validation Mean RMSE: {mean_rmse:.4f}")
        return mean_rmse
