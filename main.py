import pandas as pd
import numpy as np
import time

import os
import pandas as pd
from pandas import DataFrame
import numpy as np
import csv
from tqdm import tqdm

from preprocess import add_transformations
from experimentation import generate_analysis_graph
from utils import (
    apply_scaling,
    get_top_features_with_must_include,
    update_days_since_introduction,
)
from graphing import generate_profit_uncertainity
from basic_solver import generate_predictions
from solver_ilp import select_prices
from helpers import validation_check_price_recommendations
from all_skus_list import physical_sku_ls
import warnings

warnings.filterwarnings("ignore")
# from backtest import backtest_december, backtest_results


def setup_directories(folder_name, sub_folder_name, product_name, test_date):
    directories = [
        f"{folder_name}/Graphs/{sub_folder_name}",
        f"{folder_name}/Feature_Importances/{sub_folder_name}/{product_name}",
        f"{folder_name}/Metrics/{sub_folder_name}/{product_name}",
        f"{folder_name}/Predictions/{sub_folder_name}/{product_name}",
        f"{folder_name}/Models/{sub_folder_name}/{product_name}",
        f"{folder_name}/Inference_Graphs/{sub_folder_name}/{test_date}/{product_name}",
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_final_features(df, num_days=3):
    common_columns = [
        "Scaled_Price_Credit_A_day1",
        "Scaled_Price_Credit_A_day2",
        "Scaled_Price_Credit_A_day3",
        "price_gap",
        "Days Since Introduction",
        "diff_from_intro_price",
        "Month",
        "special_day_Elektra Day",
        "special_day_Credit Special Day",
        "Cost of Sale Credit",
        "Price To Elektra Credit (A)_day1",
        "Scaled_Price_Cash_A_day1",
    ]

    if num_days == 3:
        target_column = "y_credit_3days"
    else:
        common_columns.append("Scaled_Price_Credit_A_day4")
        target_column = "y_credit_4days"

    must_include_columns = common_columns + [target_column]
    final_features = must_include_columns + ["ds"]

    # columns_to_ignore_for_3D = ['y',
    #                             'y_3days',
    #                             'y_4days',
    #                             'ds',
    #                             'day_of_week_n',
    #                             'Brand',
    #                             'Carrier',
    #                             'Model',
    #                             'Cash Unit Sale',
    #                                 ]

    # final_features = get_top_features_with_must_include(
    #     df.drop( columns_to_ignore_for_3D, axis=1 ),
    #     df[target],
    #     must_include_columns,
    #     3
    # )

    return final_features


################# Create Transformed_Clusters

folder_name = "Experiments/Experiment 1"
model_name = "CatBoostRegressor"
sub_folder_name = model_name
test_size = 30
test_date = "2024-07-26"

sku_map = pd.read_csv("data/sku_product_name_map.csv", index_col=0)
national_price_df = pd.read_csv(
    "s3://elektra-data/commercial_comments/national_price_20240722.csv"
)
daily_data = pd.read_csv(
    "s3://elektra-data/transformed/scaled_Sanitized_Master_Physical_stores_24072024.csv"
)

all_sku_ls = sku_map["SKU"].unique()
all_models_ls = sku_map["Model"].unique()
all_brands_ls = sku_map["Brand"].unique()

daily_data = pd.merge(daily_data, sku_map, on="SKU", how="left")
df_scaled = apply_scaling(daily_data)
df_scaled["Day of Date"] = pd.to_datetime(df_scaled["Day of Date"])
df_scaled = df_scaled.drop(["level_1"], axis=1)
df_updated = update_days_since_introduction(df_scaled)
daily_data = df_updated.drop(["level_1"], axis=1)

skus_not_priced = []
count = 0

for product in tqdm(physical_sku_ls):
    count += 1
    print(f"Processing Product - {product}")
    try:
        if product in all_sku_ls:
            filtered_data = daily_data[
                (daily_data["SKU"] == product)
                & (daily_data["Pilot/Control"] == "PILOTO")
            ]
            group_by_column = "SKU"

        elif product in all_models_ls:
            filtered_data = daily_data[
                (daily_data["Model"] == product)
                & (daily_data["Pilot/Control"] == "PILOTO")
            ]
            group_by_column = "Model"

        elif product in all_brands_ls:
            filtered_data = daily_data[
                (daily_data["Brand"] == product)
                & (daily_data["Pilot/Control"] == "PILOTO")
            ]
            group_by_column = "Brand"

        product_names = f"{product}"

        filtered_data = filtered_data.drop_duplicates(subset=["SKU", "Day of Date"])

        setup_directories(
            folder_name, sub_folder_name, product_names, test_date=test_date
        )

        transformed_df = add_transformations(
            filtered_data, test_date, product_names, group_by_column=group_by_column
        )

        columns_3d = get_final_features(transformed_df, 3)
        columns_4d = get_final_features(transformed_df, 4)

        columns_3d.append(group_by_column)
        columns_4d.append(group_by_column)

        upper_bound_price = transformed_df["Scaled_Price_Credit_A_day1"].max()
        lower_bound_price = transformed_df["Scaled_Price_Credit_A_day1"].min()

        # generate_analysis_graph(
        #     features=transformed_df,
        #     columns_3d=columns_3d,
        #     columns_4d=columns_4d,
        #     product_names=product_names,
        #     folder_name=folder_name,
        #     model_name=model_name,
        #     sub_folder_name=sub_folder_name,
        #     test_size=test_size,
        #     group_by_column=group_by_column,
        # )

        date_range = pd.date_range(start=test_date, end=test_date, freq="D")

        for day in date_range:
            start_time = time.time()
            if day.weekday() == 1:
                
                transformed_df1 = transformed_df
                target = ["y_credit_3days"]

                generate_profit_uncertainity(
                    features=transformed_df1,
                    columns_3d=columns_3d,
                    model_name=model_name,
                    target=target,
                    date=day,
                    lower_bound_price=lower_bound_price,
                    upper_bound_price=upper_bound_price,
                    test_date=test_date,
                    day=day,
                    product_name=product_names,
                    folder_name=folder_name,
                    sub_folder_name=sub_folder_name,
                    group_by_column=group_by_column,
                    national_price_df=national_price_df,
                )
            elif day.weekday() == 4:
                print("date", day)

                transformed_df1 = transformed_df

                target = ["y_credit_4days"]

                generate_profit_uncertainity(
                    features=transformed_df1,
                    columns_3d=columns_4d,
                    model_name=model_name,
                    target=target,
                    date=day,
                    lower_bound_price=lower_bound_price,
                    upper_bound_price=upper_bound_price,
                    test_date=test_date,
                    day=day,
                    product_name=product_names,
                    folder_name=folder_name,
                    sub_folder_name=sub_folder_name,
                    group_by_column=group_by_column,
                    national_price_df=national_price_df,
                )
            end_time = time.time()
            execution_time = end_time - start_time

            print(f"Execution time: {execution_time} seconds")
        print(f"complete - {product} {count}")

    except Exception as e:
        skus_not_priced.append(product)
        print(f"Error: {e}")
        continue

config_select_prices = {
    "FOLDER_NAME": folder_name,
    "MODEL_NAMES": [model_name],
    "DATE": test_date,
    "SIZE": 100,
    "SKU_MAP_PATH": "data/sku_product_name_map.csv",
    "NATIONAL_PRICE_PATH": "s3://elektra-data/commercial_comments/national_price_20240722.csv",
    "OUTPUT_PATH": "Solution4.csv",
    "FINAL_OUTPUT_PATH": "data/predictions/07262024_EKT_Physical_Stores_Prices_Test.csv",
}

select_prices(config_select_prices)


commercial_comments_path = "s3://elektra-data/commercial_comments/20240722 Commercial Comments_ Price Recomendations July 22nd 2024.xlsx"
predictions_path = "data/predictions/07262024_EKT_Physical_Stores_Prices_Test.csv"

validation_check_price_recommendations(commercial_comments_path, predictions_path)

# generate_predictions(
#     dates=[test_date],
#     inference_graphs_dir=f"{folder_name}/Inference_Graphs",
#     folder_date=test_date,
#     model_names=["CatBoostRegressor"],
#     physical_sku_ls=physical_sku_ls,
# )
