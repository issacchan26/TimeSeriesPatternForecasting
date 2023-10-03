import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

input_file = "path to power_data.csv"
pd.set_option('display.max_columns', None)

def preprocess(df):
    df = df.iloc[: , 1:]
    df['Datetime'] = pd.to_datetime(df['Datetime'])  # for sorting
    df.sort_values(by=['Datetime'], inplace=True)
    df = df.reset_index(drop=True)
    df['hour'] = df['Datetime'].dt.hour
    df['year'] = df['Datetime'].dt.year
    df['month'] = df['Datetime'].dt.month
    df['day'] = df['Datetime'].dt.day
    rank_index = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2]
    hour_to_rank = dict(zip(range(0,24), rank_index))
    df['hour_rank_gp'] = df['hour'].map(hour_to_rank) 
    df['day_of_the_week'] = df['Datetime'].dt.dayofweek
    df['is_weekend'] = df['Datetime'].dt.dayofweek > 4
    m = (9 <= df['hour']) & (df['hour'] < 18) & (df['is_weekend'] == 0)
    df['business_hour'] = np.where(m, 1, 0)
    seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
    month_to_season = dict(zip(range(1,13), seasons))
    df['season'] = df['month'].map(month_to_season) 
    new_seasons = [2, 2, 1, 1, 1, 3, 3, 3, 1, 1, 1, 2]
    month_to_season = dict(zip(range(1,13), new_seasons))
    df['new_season'] = df['month'].map(month_to_season) 
    processed_df = df.iloc[: , 1:]
    return processed_df

def df_label_encoder(df):
    le = preprocessing.LabelEncoder()
    columns = df.columns.values
    for column in columns:
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            df[column] = le.fit_transform(df[column].astype(str))
    return df

def time_series_split(df, gt_col, test_ratio):
    tss = TimeSeriesSplit(n_splits=2, test_size=int(len(df)*test_ratio))
    new_df = df.drop(df[df['year']==2018].index)  # drop 2018 data
    X = new_df.drop([gt_col], axis=1)
    y = new_df[gt_col]
    X_index1, X_index2 = tss.split(X)
    train_idx, test_idx = X_index2
    X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx,:]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':

    df = pd.read_csv(input_file, on_bad_lines='skip')
    processed_df = preprocess(df)
    processed_df = df_label_encoder(processed_df)
    X_train, X_test, y_train, y_test = time_series_split(processed_df, gt_col='Power_MWH', test_ratio=0.1)
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values

    model = CatBoostRegressor(
        iterations=10000,
        task_type="GPU",
        learning_rate=0.001,
        random_seed=0,
        cat_features=[0,1,2,3,4,5,6,7,8,9]
    )

    model.fit(X_train, y_train, 
            eval_set=(X_test, y_test), 
            verbose=False
    )

    y_pred = model.predict(X_test)
    print('r2 score of CatBoost model:', r2_score(y_test, y_pred))
