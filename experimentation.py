from datetime import timedelta
import pandas as pd
import numpy as np
from models import (
    LightGBMForecaster,
    CatboostForecaster,
    XGBoostForecaster,
    RandomForestForecaster,
    GradientBoostingRegressorForecaster,
    SARIMAXForecaster,
    NegativeBinomialForecaster,
    CatBoostMultiOutputRegressorForecaster,
    # PyBATSForecaster,
)
from utils import plot_preds, save_feature_importances
import math
from tqdm import tqdm


def generate_analysis_graph(
    features,
    columns_3d,
    columns_4d,
    product_names,
    folder_name,
    model_name,
    sub_folder_name,
    test_size,
    group_by_column,
):
    
    union_set = set(columns_3d) | set(columns_4d)

    union_list = list(union_set)

    df1 = features[union_list]
    df1 = df1.dropna()

    predictions_credit = []
    predictions_cash = []
    predictions = []
    test_data = []
    dates_ls = []
    actual_credit_price_ls = []
    actual_credit_cost_ls = []

    # columns_3d.remove('y_3days')
    # columns_3d.append('y')
    # columns_4d.remove('y_4days')
    # columns_4d.append('y')

    if group_by_column == "SKU":
        product_names = int(product_names)

    max_date = pd.to_datetime(
        df1[df1[group_by_column] == product_names]["ds"]
    ).max() - timedelta(days=2)
    min_date = max_date - timedelta(days=test_size - 1)

    date_range = pd.date_range(start=min_date, end=max_date, freq="D")

    for date in date_range:
        date_obj = date.to_pydatetime()
        date_obj_bef = date_obj - timedelta(days=1)

        date_only = date_obj.date()
        date_bef_only = date_obj_bef.date()

        date = date_only.strftime("%Y-%m-%d")
        date_bef = date_bef_only.strftime("%Y-%m-%d")

        df = df1

        #############  Test only 3Day
        # df['y']= df['y_3days']
        # df = df.drop(['y_3days'],axis=1)

        # train = df[(df['ds']<date_bef)][columns_3d]
        # test = df[(df['SKU'] == product_name_int) & (df['ds']==date)][columns_3d]

        target = []
        if date_only.weekday() == 1:
            # df['y']= df['y_3days']
            # df = df.drop(['y_3days'],axis=1)

            train = df[(df["ds"] < date_bef)][columns_3d]
            # test = df[(df['ds']==date)][columns_3d]
            target = ["y_credit_3days"]

            test = df[(df[group_by_column] == product_names) & (df["ds"] == date)][
                columns_3d
            ]

        elif date_only.weekday() == 4:


            train = df[(df["ds"] < date_bef)][columns_4d]
            target = ["y_credit_4days"]

            test = df[(df[group_by_column] == product_names) & (df["ds"] == date)][
                columns_4d
            ]

        else:
            continue

        dates_ls.append(date)

        actual_credit_price = test["Price To Elektra Credit (A)_day1"].iloc[0]
        actual_credit_price_ls.append(actual_credit_price)

        actual_credit_cost = df[
            (df[group_by_column] == product_names) & (df["ds"] == date)
        ]["Cost of Sale Credit"].iloc[0]
        actual_credit_cost_ls.append(actual_credit_cost)

        if model_name == "CatBoostRegressor":
            forcaster = CatboostForecaster()
        elif model_name == "GradientBoostingRegressor":
            forcaster = GradientBoostingRegressorForecaster()
        elif model_name == "RandomForestRegressor":
            forcaster = RandomForestForecaster()
        elif model_name == "LightGBM":
            forcaster = LightGBMForecaster()
        elif model_name == "SARIMAX":
            forcaster = SARIMAXForecaster()
        elif model_name == "NegativeBinomial":
            forcaster = NegativeBinomialForecaster()
            # CatBoostMultiOutputRegressor
        elif model_name == "CatBoostMultiOutputRegressor":
            forcaster = CatBoostMultiOutputRegressorForecaster()
        # elif model_name == 'PyBATSForecaster':
        #     forcaster = PyBATSForecaster()
        else:
            forcaster = XGBoostForecaster()
        columns_to_drop = target + [
            "ds",
            "Cost of Sale Credit",
            "Price To Elektra Credit (A)_day1",
            group_by_column,
        ]
        X_train, y_train = train.drop(columns_to_drop, axis=1), train[target]
        X_test, y_test = test.drop(columns_to_drop, axis=1), test[target]

        forcaster.fit(y_train, X_train)
        y_pred = forcaster.predict(X_test)
        # predictions_cash = 0

        predictions_credit = y_pred[0]

        actual_demand_day_level = sum(y_test.values)

        actual_demand_day_level = actual_demand_day_level[0]

        predicted_demand_day_level = max(round(predictions_credit), 0)
        predictions.append(predicted_demand_day_level)
        test_data.append(actual_demand_day_level)

        print("pred", predicted_demand_day_level)
        print("actual", actual_demand_day_level)
        print("complete - ", date)

    model_path = f"{folder_name}/Models/{sub_folder_name}/" f"{product_names}.pkl"

    title = f"{product_names}"

    dates_datetime = pd.to_datetime(dates_ls)

    datetime_index = pd.DatetimeIndex(dates_datetime)

    plot_preds(
        datetime_index,
        features,
        predictions,
        title=title,
        end_val=test_size,
        folder=sub_folder_name,
        parent_folder_name=folder_name,
        model_name=model_name,
        actual=test_data,
        actual_credit_price_ls=actual_credit_price_ls,
        actual_credit_cost_ls=actual_credit_cost_ls,
        group_by_column=group_by_column,
    )
    save_feature_importances(forcaster, folder_name, sub_folder_name, title)
