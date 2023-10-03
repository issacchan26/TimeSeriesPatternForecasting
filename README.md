# TimeSeriesPatternForecasting
Time series forecasting on power consumption pattern with CatBoost and regression model

## Dataset
There are two columns in the [power_data.csv](./power_data.csv): timestamp and hourly power consumption.  
The goal of the model is to predict the pattern of power consumption, e.g. weekly, monthly, and seasonal patterns.

## Data analysis and visualization
The detail is provided in the notebook, please change the path of power_data.csv  
The notebook provide data visualization and model building pipeline with a baseline regression model and CatBoost model

## Final model
The script of final model is saved in [model.py](./model.py)
