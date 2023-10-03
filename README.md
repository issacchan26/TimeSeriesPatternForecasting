# TimeSeriesPatternForecasting
Time series forecasting on power consumption pattern with CatBoost and regression model

## Dataset
There are two columns in the [power_data.csv](./power_data.csv): timestamp and hourly power consumption.  
The goal of the model is to predict the pattern of power consumption, e.g. weekly, monthly, and seasonal patterns.

## Data analysis and visualization
The data analysis, model comparison and selection, model building and evaluation, conclusion and recommendation are provided in the [notebook](./data_analysis_visualization.ipynb)

## Final model
The script of final model is saved in [model.py](./model.py)  
R-squared score is used to evaluate our regression models.
