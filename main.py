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

import warnings

warnings.filterwarnings("ignore")
# from backtest import backtest_december, backtest_results


def setup_directories(folder_name, sub_folder_name, product_name, test_date):
    os.makedirs(f"{folder_name}/Graphs/{sub_folder_name}", exist_ok=True)
    os.makedirs(
        f"{folder_name}/Feature_Importances/{sub_folder_name}/{product_name}",
        exist_ok=True,
    )
    os.makedirs(
        f"{folder_name}/Metrics/{sub_folder_name}/{product_name}", exist_ok=True
    )
    os.makedirs(
        f"{folder_name}/Predictions/{sub_folder_name}/{product_name}", exist_ok=True
    )
    os.makedirs(f"{folder_name}/Models/{sub_folder_name}/{product_name}", exist_ok=True)
    os.makedirs(
        f"{folder_name}/Inference_Graphs/{sub_folder_name}/{test_date}/{product_name}",
        exist_ok=True,
    )


def get_final_features(df, num_days=3):
    if num_days == 3:
        # target= 'y_3days'
        must_include_columns = [
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

            "y_credit_3days",
            
            # "Scaled_Price_Credit_A_day1",
            # "Scaled_Price_Credit_A_day2",
            # "Scaled_Price_Credit_A_day3",
            # "Scaled_Price_Cash_A_day1",
            # "Scaled_Price_Cash_A_day2",
            # "Scaled_Price_Cash_A_day3",
            # "Price To Elektra Credit (A)_day1",
            # # "Scaled_Price_Credit_DE_day1",
            # "Cost of Sale Credit",
            # "price_gap",
            # # "time_price_interaction",
            # #    'day_of_week_n',
            # #    'SKU',
            # # 'Model',
            # #    'if_payday',
            # # "special_day_Credit Special Day",
            # "Days Since Introduction",
            # "diff_from_intro_price",
            # # 'ratio_sales_3days_lag5',
            # "rolling_mean_3days_credit_lag7",
            # "y_credit_max_3days_last2weeks",
            # # 'discount_ratio_mean_3days',
            # "Month",
            # #    'Scaled_Price_Credit_DE_mean_3days',
            # "special_day_Elektra Day",
            # "special_day_Credit Special Day",
            # "y_credit_3days",
        ]
    else:
        # target= 'y_4days'
        must_include_columns = [
            "Scaled_Price_Credit_A_day1",
            "Scaled_Price_Credit_A_day2",
            "Scaled_Price_Credit_A_day3",
            "Scaled_Price_Credit_A_day4",
            "price_gap",
            "Days Since Introduction",
            "diff_from_intro_price",
            "Month",
            "special_day_Elektra Day",
            "special_day_Credit Special Day",

            "Cost of Sale Credit",
            "Price To Elektra Credit (A)_day1",
            "Scaled_Price_Cash_A_day1",

            "y_credit_4days",



            # "Scaled_Price_Credit_A_day1",
            # "Scaled_Price_Credit_A_day2",
            # "Scaled_Price_Credit_A_day3",
            # "Scaled_Price_Credit_A_day4",
            # "Scaled_Price_Cash_A_day1",
            # "Scaled_Price_Cash_A_day2",
            # "Scaled_Price_Cash_A_day3",
            # "Scaled_Price_Cash_A_day4",
            # # "Scaled_Price_Credit_DE_day1",
            # "Cost of Sale Credit",
            # "Price To Elektra Credit (A)_day1",
            # "price_gap",
            # # "time_price_interaction",
            # #    'Scaled_Price_Credit_DE_mean_4days',
            # #    'Scaled_Price_Credit_A_mean_4days' ,
            # #     'Scaled_Price_Cash_A_mean_4days' ,
            # "day_of_week_n",
            # #    'SKU',
            # "diff_from_intro_price",
            # "Days Since Introduction",
            # #    'if_payday',
            # #    'ratio_sales_4days_lag6',
            # "Month",
            # "rolling_mean_4days_credit_lag7",
            # # 'discount_ratio_mean_4days',
            # # 'product_trend_4days_lag5',
            # "y_credit_max_4days_last2weeks",
            # # 'product_seasonal_4days_lag5',
            # "special_day_Credit Special Day",
            # "special_day_Elektra Day",
            # #    'rolling_mean_4days_cash_lag7',
            # #    'Model',
            # "y_credit_4days",
        ]

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

    final_features = must_include_columns
    final_features.append("ds")

    return final_features


physical_sku_ls = [
    31057271,
    31057273,
    31057286,
    31057163,
    31057164,
    31057172,
    31055497,
    31055496,
    31058953,
    31058928,
    31055495,
    31057162,
    31057150,
    31057161,
    31056539,
    31056541,
    31056540,
    31058721,
    31053498,
    31053497,
    31058051,
    31058056,
    31059004,
    31058997,
    31057958,
    31057948,
    31057959,
    31057947,
    31058023,
    31058018,
    31057257,
    31057258,
    31057263,
    31059186,
    31059202,
    31057715,
    31057707,
    31058923,
    31058932,
    31058942,
    31058922,
    31057048,
    31057047,
    31058674,
    31058675,
    31058672,
    31058673,
    31058676,
    31058639,
    31058640,
    31058643,
    31058652,
    31058642,
    31058659,
    31057523,
    31058048,
    31058054,
    31055524,
    31055523,
    31058887,
    31058869,
    31058008,
    31058000,
    31058005,
    31051762,
    31059222,
    31058905,
    31058894,
    31058030,
    31058047,
    31057579,
    31057592,
    31057589,
    31057590,
    31057580,
    31057561,
    31057581,
    31057519,
    31057661,
    31057680,
    31057666,
    31058082,
    31058078,
    31058071,
    31057343,
    31057337,
    31057311,
    31057400,
    31057384,
    31058847,
    31058968,
    31058984,
    31057087,
    31057088,
    31058993,
    31058979,
    31058988,
    31056545,
    31056544,
    31055343,
    31058034,
    31058028,
    31059240,
    31059263,
    31058767,
    31058766,
    31058087,
    31058094,
    31058971,
    31058992,
    31057241,
    31057247,
    31058989,
    31058986,
    31057239,
    31057245,
    31059073,
    31059068,
    31057246,
    31057240,
    31057236,
    31057249,
    31059119,
    31059074,
    31058033,
    31058027,
    31059260,
    31058710,
    31058754,
    31058032,
    31058025,
    31059261,
    31059241,
    31058896,
    31058723,
    31057753,
    31057743,
    31058022,
    31059139,
    31059203,
    31057767,
    31057776,
    31058977,
    31057732,
    31055552,
    31055554,
    31055383,
    31045304,
    31050860,
    31055362,
    31047746,
    31055363,
    31032647,
    31052376,
    31055352,
    31052384,
    31055310,
    31057260,
    31057283,
    31050784,
    31050785,
    31051900,
    31051907,
    31051954,
    31051955,
    31051906,
    31051965,
    31051962,
    31051898,
    31054964,
    31054965,
    31055603,
    31055604,
    31052267,
    31049642,
    31050805,
    31055373,
    31055374,
    31057800,
    31057807,
    31057811,
    31051491,
    31051512,
    31057272,
    31057285,
    31050797,
    31050798,
    31048023,
    31050806,
    31048290,
    31051504,
    31051505,
    31041272,
    31053174,
    31053175,
    31045254,
    31045268,
    31056551,
    31056552,
    31056553,
    31023696,
    31023697,
    31044891,
    31047401,
    31050116,
    31047968,
    31047969,
    31044050,
    31048641,
    31055521,
    31055526,
    31055692,
    31055702,
    31045301,
    31054264,
    31054272,
    31056520,
    31056530,
    31055129,
    31052285,
    31057217,
    31057223,
    31044759,
    31051425,
    31051409,
    31051415,
    31051929,
    31051937,
    31050449,
    31055509,
    31055510,
    31055530,
    31055531,
    31055542,
    31051024,
    31057153,
    31057154,
    31057155,
    31057861,
    31057871,
    31057872,
    31047852,
    31047853,
    31047854,
    31050996,
    31050998,
    31044773,
    31044774,
    31051032,
    31051033,
    31051034,
    31051035,
    31044766,
    31051002,
    31051004,
    31051005,
    31058937,
    31058951,
    31058952,
    31051597,
    31051599,
    31051600,
    31050458,
    31057598,
    31057607,
    31051973,
    31051974,
    31051941,
    31051982,
    31055766,
    31055768,
    31057908,
    31057912,
    31055765,
    31055767,
    31057801,
    31057809,
    31050186,
    31051148,
    31055200,
    31052400,
    31054970,
    31052120,
    31052121,
    31057256,
    31057270,
    31051165,
    31051038,
    31057259,
    31057264,
    31047788,
    31043249,
    31050114,
    31050115,
    31050108,
    31050109,
    31050111,
    31050112,
    31057252,
    31047888,
    31047886,
    31057267,
    31051481,
    31051499,
    31051501,
    31057237,
    31057250,
    31057765,
    31052027,
    31052028,
    31051547,
    31055644,
    31055645,
    31051424,
    31049136,
    31051523,
    31051524,
    31055599,
    31055600,
    31050260,
    31050261,
    31051093,
    31051094,
    31051434,
    31051435,
    31048066,
    31053162,
    31053163,
    31048650,
    31055757,
    31055758,
    31044904,
    31047799,
    31047800,
    31055314,
    31055315,
    31055029,
    31052398,
    31052399,
    31056536,
    31056537,
    31057637,
    31057594,
    31057603,
    31053525,
    31053526,
    31054866,
    31054867,
    31054868,
    31047890,
    31051654,
    31051655,
    31051656,
    31051694,
    31051695,
    31055563,
    31055564,
    31055565,
    31059108,
    31059122,
    31059109,
    31059120,
    31057248,
    31057262,
    31045265,
    31057238,
    31057244,
    31052116,
    31052118,
    31057745,
    31059146,
    31055731,
    31055748,
    31057742,
    31057752,
    31055749,
    31055742,
    31055750,
    31030392,
    31056563,
    31056564,
    31055735,
    31055752,
    31055745,
    31057734,
    31057711,
    31057725,
    31057735,
    31057909,
    31057913,
    31057916,
    31057953,
    31057768,
    31057777,
    31049532,
    31049533,
    31048227,
    31036667,
    31050594,
    31050595,
    31056360,
    31057714,
    31055490,
    31057691,
    31057695,
    31058868,
    31058879,
    31055416,
    31055417,
    31055630,
    31055631,
    31052240,
    31052241,
    31051678,
    31051542,
    31051543,
    31055194,
    31048002,
    31048298,
    31048299,
    31050521,
    31050525,
    31050534,
    31050536,
    31053203,
    31053204,
    31050968,
    31050971,
    31050976,
    31050977,
    31047427,
    31048389,
    31055895,
    31055896,
    31055016,
    31055017,
    31055108,
    31055109,
    31058009,
    31058015,
    31052438,
    31052440,
    31057018,
    31057019,
    31047340,
    31051356,
    31051357,
    31048865,
    31048866,
    31056272,
    31056273,
    31056274,
    31056635,
    31056636,
    31056637,
    31056262,
    31057206,
    31057212,
    31057213,
    31057613,
    31057625,
    31057632,
    31051838,
    31051839,
    31051840,
    31052155,
    31052156,
    31045043,
    31050179,
    31057652,
    31057737,
    31052082,
    31052083,
    31047831,
    31052091,
    31052112,
    31056959,
    31056960,
    31057899,
    31057907,
    31048828,
    31048830,
    31057027,
    31057028,
    31050297,
    31055158,
    31055159,
    31055160,
    31051402,
    31053437,
    31053438,
    31053439,
    31057356,
    31045187,
    31059053,
    31050323,
    31056406,
    31050360,
    31057403,
    31057419,
    31057425,
    31058720,
    31057432,
    31057447,
    31054895,
    31054896,
    31054897,
    31054923,
    31057441,
    31057463,
    31057856,
    31057870,
]


top_products = [
    31057273, 31056539, 31057286, 31057271, 31055735, 31056541, 31057114, 31057115, 
    31057122, 31059500, 31055496, 31048298, 31051094, 31059489, 31059514, 31058690, 
    31058754, 31057111, 31059397, 31057753, 31058710, 31057807, 31058048, 31058063, 
    31055509, 31056406, 31057272, 31057958, 31058062, 31055599, 31055752, 31056540, 
    31059509, 31059942, 31055630, 31056273, 31058896, 31055631, 31058689
]

################# Create Transformed_Clusters

skus_map = pd.read_csv("data/sku_product_name_map.csv")

all_sku_ls = skus_map["SKU"].unique()
all_models_ls = skus_map["Model"].unique()
all_brands_ls = skus_map["Brand"].unique()

temp = [31051762]
skus_not_priced = []
count = 0

national_price_map = pd.read_csv("data/national_price_20240722.csv")

daily_data = pd.read_csv(
            "s3://elektra-data/transformed/scaled_Sanitized_Master_Physical_stores_24072024.csv"
        )
sku_map = pd.read_csv("data/sku_product_name_map.csv", index_col=0)

daily_data = pd.merge(daily_data, sku_map, on="SKU", how="left")

df_scaled = apply_scaling(daily_data)
df_scaled["Day of Date"] = pd.to_datetime(df_scaled["Day of Date"])
df_scaled = df_scaled.drop(["level_1"], axis=1)

df_updated = update_days_since_introduction(df_scaled)
daily_data = df_updated.drop(["level_1"], axis=1)

for product in tqdm(top_products):
    count += 1
    print('Product - ',product)
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
        folder_name = "Experiments/Experiment 1"
        model_name = "CatBoostRegressor"
        sub_folder_name = model_name
        test_size = 30
        test_date = "2024-07-26"

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

        date_range = pd.date_range(start=test_date, end=test_date, freq='D')

        for day in date_range:
            start_time = time.time()
            if day.weekday() == 1:
                # print('date',day)

                transformed_df1 = transformed_df
                target =  ['y_credit_3days']
                

                generate_profit_uncertainity(features=transformed_df1, columns_3d=columns_3d,
                                            model_name=model_name, target=target,
                                            date=day,
                                                lower_bound_price=lower_bound_price,
                                        upper_bound_price=upper_bound_price,
                                        test_date=test_date,
                                        day=day,
                                        product_name=product_names,
                                        folder_name=folder_name,
                                        sub_folder_name=sub_folder_name,
                                        group_by_column = group_by_column)
            elif day.weekday() == 4:
                print('date',day)

                transformed_df1 = transformed_df

                target =  ['y_credit_4days']

                generate_profit_uncertainity(features=transformed_df1, columns_3d=columns_4d,
                                            model_name=model_name, target=target,
                                            date=day,
                                                lower_bound_price=lower_bound_price,
                                        upper_bound_price=upper_bound_price,
                                        test_date=test_date,
                                        day=day,
                                        product_name=product_names,
                                        folder_name=folder_name,
                                        sub_folder_name=sub_folder_name,
                                        group_by_column = group_by_column)
            end_time = time.time()
            execution_time = end_time - start_time



            print(f"Execution time: {execution_time} seconds")
        print(f"complete - {product} {count}")

    except Exception as e:
        skus_not_priced.append(product)
        print(f"Error: {e}")
        continue


# generate_predictions(
#     dates=[test_date],
#     inference_graphs_dir=f"{folder_name}/Inference_Graphs",
#     folder_date=test_date,
#     model_names=["CatBoostRegressor"],
#     physical_sku_ls=physical_sku_ls,
# )

filename = "June7_not_priced_digital.csv"

with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    for element in skus_not_priced:
        writer.writerow([element])

print(f"Data has been written to {filename}")
