from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ARDRegression, ElasticNet, BayesianRidge

# %pip install statsmodels
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson, NegativeBinomial

from statsmodels.tsa.statespace.sarimax import SARIMAX


# from pybats.analysis import analysis
# from pybats.seasonal import seascomp
# from pybats.dglm import dglm
# from pybats.forecast import forecast



class TimeSeriesForecaster(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class PipelineOptimizer(ABC):
    @abstractmethod
    def optimize(self, X, y):
        pass

# class PyBatsForecaster(TimeSeriesForecaster):
#     def __init__(self):
#         self.model_name = "PyBatsForecaster"
#         self.forecast_samps = 2000
#         self.k = 7
#         self.rho = 0.5
#         self.prior_length = 21
#         self.model = None
#         self.samples = None
#         self.model_coef = None

#     def fit(self, y, regressors, dates):
#         # Convert your data to the format required by pybats
#         y = y.values
#         regressors = regressors.values

#         # Assuming regressors and dates are provided
#         forecast_start = dates[-1] + pd.Timedelta(days=1)  
#         forecast_end = forecast_start + pd.Timedelta(days=30)  

#         self.model, self.samples, self.model_coef = analysis(
#             y, 
#             regressors,
#             self.k, 
#             forecast_start, 
#             forecast_end, 
#             nsamps=self.forecast_samps,
#             family='poisson',
#             seasPeriods=[7], 
#             seasHarmComponents=[[1,2,3]],
#             prior_length=self.prior_length, 
#             dates=dates, 
            
#             rho=self.rho,
#             ret=['model', 'forecast', 'model_coef']
#         )
#         return pd.DataFrame()

#     def predict(self, X):
#         # Perform prediction using the fitted PyBats model
#         if self.model is None:
#             raise Exception("Model has not been fit yet.")

#         forecast_start = X.index[0]
#         forecast_end = X.index[-1]

#         # Perform forecasting
#         _, samples, _ = analysis(
#             self.model.y, 
#             X.values, 
#             self.k, 
#             forecast_start, 
#             forecast_end, 
#             nsamps=self.forecast_samps, 
#             family='poisson',
#             seasPeriods=[7], 
#             seasHarmComponents=[[1,2,3]],
#             prior_length=self.prior_length, 
#             dates=pd.date_range(forecast_start, forecast_end, freq='D'), 
            
#             rho=self.rho,
#             ret=['forecast']
#         )

#         # Aggregate samples to get mean forecast
#         forecast_mean = np.mean(samples, axis=0)
#         return list(forecast_mean)

# class ProphetForecaster(TimeSeriesForecaster):
#     def __init__(self):
#         self.model_name = 'Prophet'
#         self.params_grid = {
#             'seasonality_mode': 'multiplicative',
#             'daily_seasonality': True,
#             'weekly_seasonality': True,
#             'yearly_seasonality': True,
#         }
#         # Prophet doesn't use a lags_grid like the other models,
#         # but you can keep it for consistency if you want.
#         self.lags_grid = [7]
#         self.model = Prophet(**self.params_grid)

#     def fit(self, y, regressors=None):
#         # Prepare the data in the format Prophet expects
#         data = pd.DataFrame({'ds': y.index, 'y': y.values})
#         # Add regressors if provided
#         if regressors is not None:
#             data = data.join(regressors)
#             for col in regressors.columns:
#                 self.model.add_regressor(col)
#         # Fit the model
#         self.model.fit(data)
#         return pd.DataFrame()  # Return empty DataFrame or maybe some fitting info

#     def predict(self, X):
#         # Prepare future DataFrame
#         future = self.model.make_future_dataframe(periods=len(X))
#         # Add regressors if provided
#         if X is not None:
#             future = future.join(X)
#         # Make predictions
#         forecast = self.model.predict(future)
#         return forecast['yhat']  # Return predictions


class LightGBMForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = "LightGBM"
        self.params_grid = {
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": 5,
            "num_leaves": 32,
            "min_data_in_leaf": 20,
            "objective": "poisson",
            "verbosity": -1,
            "silent": True,
        }
        self.lags_grid = [7]
        self.model = lgb.LGBMRegressor(**self.params_grid, random_state=100, verbose=0)

    def fit(self, y, regressors):
        # The grid search forecaster comes here
        self.model.fit(regressors, y)
        return pd.DataFrame()

    def predict(self, X):
        # Perform prediction using the fitted LightGBM model
        return self.model.predict(X)


class SARIMAXForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = "SARIMAX"
        # SARIMAX parameters: p, d, q for ARIMA and P, D, Q, s for seasonal component
        # self.params_grid = dict(
        #     p=[1, 2, 3],
        #     d=[0, 1],
        #     q=[1, 2, 3],
        #     P=[0, 1, 2],
        #     D=[0, 1],
        #     Q=[0, 1, 2],
        #     s=[12]  # assuming a yearly seasonal component
        # )
        self.lags_grid = [7]  # Example lag grid
        self.model = None  # Model will be initialized in fit method

    def fit(self, y, regressors):
        # The grid search forecaster can be implemented here
        # For simplicity, we'll use p=1, d=1, q=1, P=1, D=1, Q=1, s=12
        self.model = SARIMAX(
            y,
            exog=regressors,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_fit = self.model.fit()
        return pd.DataFrame()  # return some DataFrame if necessary

    def predict(self, X):
        # Perform prediction using the fitted SARIMAX model
        # The 'steps' parameter specifies how many steps in the future to forecast
        forecast = self.model_fit.forecast(steps=len(X), exog=X)
        return forecast


class GradientBoostingRegressorForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = "GradientBoostingRegressor"
        self.params_grid = dict(
            learning_rate=0.05,
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=9,
            min_samples_split=9,
        )
        self.lags_grid = [7]
        self.model = GradientBoostingRegressor(
            loss="squared_error", **self.params_grid, random_state=100
        )

    def fit(self, y, regressors):
        # The grid search forcaster comes here
        self.model.fit(regressors, y)
        return pd.DataFrame()

    def predict(self, X):
        # Perform prediction using the fitted GradientBoosting model
        return self.model.predict(X)


class CatboostForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = "CatBoostRegressor"
        # self.params_grid = {
        #     'n_estimators': 100,
        #     'max_depth': 5,
        #     'learning_rate': 0.1
        #     }
        # self.lags_grid = [7]
        self.model = CatBoostRegressor(verbose=0, random_seed=100)
        # loss_function="RMSEWithUncertainty", **self.params_grid)

    def fit(self, y, regressors):
        # The grid search forcaster comes here
        
        recency_weights = np.arange(1, len(regressors) + 1) ** 0.95
        train_pool = Pool(data=regressors, label=y, weight=recency_weights)
        self.model.fit(train_pool)
        return pd.DataFrame()

    def predict(self, X):
        # Perform prediction using the fitted ARIMA model
        return list(self.model.predict(X))



# class BSTSForecaster:
#     def __init__(self):
#         self.model_name = "BSTS"
#         self.model = None
#         self.scaler = StandardScaler()

#     def fit(self, y, regressors):
#         # Scaling the regressors
#         scaled_regressors = self.scaler.fit_transform(regressors)

#         # Create the BSTS model
#         self.model = BSTS(
#             y=y,
#             X=scaled_regressors,
#             niter=1000,  # Number of iterations for MCMC
#             expected_model_size=10,  # Adjust based on your model complexity
#             model_components={"trend": True, "seasonal": True, "regression": True},
#         )
#         self.model.fit()
#         return pd.DataFrame()

#     def predict(self, X):
#         # Scaling the input data
#         scaled_X = self.scaler.transform(X)

#         # Perform prediction using the fitted BSTS model
#         forecast = self.model.predict(scaled_X)
#         return forecast["mean"].tolist()


class RandomForestForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = "RandomForestRegressor"
        self.params_grid = {"n_estimators": 50, "max_depth": 5}
        self.lags_grid = [7]
        self.model = RandomForestRegressor(**self.params_grid, random_state=100)

    def fit(self, y, regressors):
        # Log-transform the target variable
        y_log = np.log1p(y)
        # Train the model on the transformed target
        self.model.fit(regressors, y_log)
        return pd.DataFrame()

    def predict(self, X):
        # Predict using the trained RandomForest model on the log-transformed target
        log_predictions = self.model.predict(X)
        # Transform the predictions back to the original scale
        predictions = np.expm1(log_predictions)

        # predictions = np.clip(predictions, 0, None)
        return predictions


class XGBoostForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = "XGBoost"
        self.params_grid = {
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": 5,
            "min_child_weight": 1,
            "gamma": 0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            # 'objective': 'reg:squaredlogerror',
        }
        self.lags_grid = [7]
        self.model = xgb.XGBRegressor(random_state=100, objective="reg:squarederror")

    def fit(self, y, regressors):
        # The grid search forecaster comes here
        self.model.fit(regressors, y)
        return pd.DataFrame()

    def predict(self, X):
        # Perform prediction using the fitted XGBoost model
        return self.model.predict(X)


class NegativeBinomialForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = "NegativeBinomial"
        # No explicit hyperparameters grid for this model in the example
        # but you can expand on this if needed
        self.lags_grid = [7]  # Example lag grid
        self.model = None  # Model will be initialized in fit method
        self.results = None

    def fit(self, y, regressors):
        # Fit the model using the Negative Binomial family
        self.model = GLM(y, regressors, family=NegativeBinomial())
        self.results = self.model.fit()
        return pd.DataFrame()  # return some DataFrame if necessary

    def predict(self, X):
        # Make predictions using the fitted model
        predictions = self.results.predict(X)
        return predictions


class CatboostWithUncertainityForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = "CatBoostRegressorWithUncertainity"
        # self.params_grid = {
        # #     'n_estimators': 100,
        # #     'max_depth': 5,
        # #     'learning_rate': 0.1
        # #     }
        self.model = CatBoostRegressor(
            verbose=0,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_seed=100,
            loss_function="RMSEWithUncertainty",
        )

    def fit(self, y, regressors):
        self.model.fit(regressors, y)

    def predict(self, X):
        predictions_with_uncertainty = self.model.predict(X)
        predictions, uncertainties = (
            predictions_with_uncertainty[:, 0],
            predictions_with_uncertainty[:, 1],
        )
        return predictions, uncertainties


class CatBoostMultiOutputRegressorForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = "CatBoostMultiOutputRegressor"
        self.model = CatBoostRegressor(
            loss_function="MultiRMSE",
            verbose=0,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_seed=100,
        )

    def fit(self, y, regressors):
        self.model.fit(regressors, y)

    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions


class ElasticNetForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = "ElasticNet"
        self.params_grid = {
            "alpha": [0.1, 0.5, 1, 2],  # Regularization strength
            "l1_ratio": [0.2, 0.5, 0.8],  # Mix ratio between L1 and L2 regularization
            "max_iter": [1000, 5000],  # Maximum number of iterations
        }
        self.lags_grid = [7]
        self.model = ElasticNet(random_state=100)

    def fit(self, y, regressors):
        # Using GridSearchCV to find the best parameters
        gridsearch = GridSearchCV(self.model, self.params_grid, cv=5)
        gridsearch.fit(regressors, y)
        self.model = gridsearch.best_estimator_
        return pd.DataFrame({"Best Params": gridsearch.best_params_})

    def predict(self, X):
        # Perform prediction using the fitted ElasticNet model
        return self.model.predict(X)


class ARDRegressionForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = "ARDRegression"
        self.params_grid = {
            "n_iter": [300, 500],  # Number of iterations
            "alpha_1": [
                1e-6,
                1e-5,
            ],  # Hyperparameter of the Gamma distribution prior over the alpha parameter
            "alpha_2": [
                1e-6,
                1e-5,
            ],  # Hyperparameter of the Gamma distribution prior over the alpha parameter
            "lambda_1": [
                1e-6,
                1e-5,
            ],  # Hyperparameter of the Gamma distribution prior over the lambda parameter
            "lambda_2": [
                1e-6,
                1e-5,
            ],  # Hyperparameter of the Gamma distribution prior over the lambda parameter
        }
        self.lags_grid = [7]
        self.model = ARDRegression()

    def fit(self, y, regressors):
        # Fit the ARDRegression model
        self.model.fit(regressors, y)
        return pd.DataFrame({"Parameters": [self.model.get_params()]})

    def predict(self, X):
        # Perform prediction using the fitted ARDRegression model
        return self.model.predict(X)


class BayesianRidgeForecaster(TimeSeriesForecaster):
    def __init__(self):
        self.model_name = "BayesianRidge"
        self.model = BayesianRidge()

    def fit(self, y, regressors):
        # Fit the Bayesian Ridge Regression model
        self.model.fit(regressors, y)
        return pd.DataFrame({"Parameters": [self.model.get_params()]})

    def predict(self, X):
        # Perform prediction using the fitted Bayesian Ridge model
        return self.model.predict(X)


# class DampedLinearTrendForecaster(TimeSeriesForecaster):
#     def __init__(self):
#         self.model_name = 'DampedLinearTrend'
#         self.model = DLT(
#             response_col='y',
#             date_col='ds',
#             seasonality=24,
#             seed=42
#         )
#         self.results = None

#     def fit(self, y, regressors):
#         data = pd.concat([y, regressors], axis=1)
#         data.columns = ['y'] + list(regressors.columns)

#         if 'ds' not in data.columns:
#             data['ds'] = pd.date_range(start='2020-01-01', periods=len(data), freq='D')

#         self.model.fit(df=data)
#         self.results = self.model
#         return pd.DataFrame()

#     def predict(self, X):

#         if 'ds' not in X.columns:
#             X['ds'] = pd.date_range(start='2020-01-01', periods=len(X), freq='D')

#         predictions = self.results.predict(df=X)
#         return list(predictions['prediction'])


# %pip install tensorflow
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import pandas as pd
# import numpy as np

# class NeuralNetworkForecaster(TimeSeriesForecaster):
#     def __init__(self):
#         self.model_name = 'NeuralNetwork'
#         # Hyperparameters can be adjusted or extended
#         self.params_grid = {
#             'learning_rate': [0.001, 0.01],
#             'epochs': [50, 100],
#             'batch_size': [32, 64],
#             'layers': [(64, 32), (128, 64, 32)]  # Example: 2 or 3 layers with different units
#         }
#         self.lags_grid = [7]  # This can be adjusted based on your time series structure
#         self.model = None  # Model will be built in the fit method

#     def _build_model(self, input_shape, layers):
#         model = Sequential()
#         for units in layers:
#             model.add(Dense(units, activation='relu', input_shape=(input_shape,)))
#         model.add(Dense(1))  # Output layer for regression
#         model.compile(optimizer='adam', loss='mean_squared_error')
#         return model

#     def fit(self, X, y):
#         # Build and fit the neural network model
#         input_shape = X.shape[1]  # Assuming X is a 2D array with shape (n_samples, n_features)
#         # Example of selecting the first combination of hyperparameters
#         params = self.params_grid
#         self.model = self._build_model(input_shape, params['layers'][0])
#         self.model.fit(X, y, epochs=params['epochs'][0], batch_size=params['batch_size'][0], verbose=1)
#         return pd.DataFrame({'Status': ['Model Trained']})

#     def predict(self, X):
#         # Perform prediction using the fitted neural network model
#         predictions = self.model.predict(X)
#         return np.ravel(predictions)  # Flatten the predictions array
