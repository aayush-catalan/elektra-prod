#  New code
import math
import plotly.graph_objects as go
import pandas as pd
from datetime import timedelta
import numpy as np

from models import (
    LightGBMForecaster,
    CatboostForecaster,
    XGBoostForecaster,
    CatBoostMultiOutputRegressorForecaster,
    CatboostWithUncertainityForecaster,
)

from utils import (
    plot_uncertainity_revenue_graph,
    plot_uncertainity_demand_graph,
    plot_3d_volume,
    plot_3d_Revenue,
)

# def scale_back_prices(sku, scaled_price, first_price_map, scale_factor=1000):
#     if sku in first_price_map:
#         original_price = (scaled_price / scale_factor) * first_price_map[sku]
#         return original_price
#     else:
#         raise ValueError(f"SKU {sku} not found in the first price map.")


def scale_back_prices(
    criterion_value, criterion_type, scaled_price, first_price_df, scale_factor=1000
):
    if criterion_type not in ["SKU", "Model", "Brand"]:
        raise ValueError(
            f"Invalid criterion type {criterion_type}. Must be one of ['SKU', 'Model', 'Brand']."
        )

    if criterion_value in first_price_df[criterion_type].values:
        first_price = first_price_df[
            first_price_df[criterion_type] == criterion_value
        ].iloc[0]["Introductory_price"]
        original_price = (scaled_price / scale_factor) * first_price
        return original_price
    else:
        raise ValueError(f"{criterion_type} {criterion_value} not found in the data.")


def generate_profit_uncertainity(
    features,
    columns_3d,
    target,
    model_name,
    date,
    lower_bound_price,
    upper_bound_price,
    test_date,
    day,
    product_name,
    folder_name,
    sub_folder_name,
    group_by_column,
):
    df = features[columns_3d]

    if group_by_column == "SKU":
        product_name = int(product_name)

    df = df.dropna()

    date_obj = date.to_pydatetime()
    date_obj_bef = date_obj - timedelta(days=1)
    date = date_obj.strftime("%Y-%m-%d")
    date_bef = date_obj_bef.strftime("%Y-%m-%d")

    if df[df["ds"] == date].empty:
        return

    train = df[df["ds"] < (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")][
        columns_3d
    ]

    # if model_name == 'CatBoostRegressor':
    #     forcaster = CatboostForecaster()
    # elif model_name == 'LightGBM':
    #     forcaster = LightGBMForecaster()
    # elif model_name == 'CatBoostRegressorWithUncertainity':
    #     forcaster = CatboostWithUncertainityForecaster()
    # elif model_name == 'CatBoostMultiOutputRegressor':
    #         forcaster = CatBoostMultiOutputRegressorForecaster()
    # else:
    #     forcaster = XGBoostForecaster()

    model_mapping = {
        "CatBoostRegressor": CatboostForecaster,
        "LightGBM": LightGBMForecaster,
        "CatBoostRegressorWithUncertainity": CatboostWithUncertainityForecaster,
        "CatBoostMultiOutputRegressor": CatBoostMultiOutputRegressorForecaster,
        "XGBoostForecaster": XGBoostForecaster,
    }

    forcaster = model_mapping.get(model_name, XGBoostForecaster)()

    columns_to_drop = target + ["ds", "Cost of Sale Credit", "Scaled_Price_Cash_A_day1", group_by_column]

    X_train, y_train = train.drop(columns_to_drop, axis=1), train[target]

    forcaster.fit(y_train, X_train)

    precision = 3

    sku_map = pd.read_csv("data/sku_product_name_map.csv")
    first_price = sku_map[sku_map[group_by_column] == product_name].iloc[0][
        "Introductory_price"
    ]
    scaling_factor = 1000 / first_price

    national_price_df = pd.read_csv("data/national_price_20240722.csv", index_col=0)

    product_national_price = national_price_df[
        national_price_df[group_by_column] == product_name
    ].iloc[0]
    national_price_creditA = product_national_price["Control Credit Precio A"]
    national_price_cashA = product_national_price["Control Cash Precio A"]

    scaled_national_price_credit_A = national_price_creditA * scaling_factor
    scaled_national_price_cash_A = national_price_cashA * scaling_factor

    lower_bound_price_credit = scaled_national_price_credit_A * 0.85
    upper_bound_price_credit = scaled_national_price_credit_A * 1.15

    desired_intervals = 100
    range_span_credit = upper_bound_price_credit - lower_bound_price_credit
    step_size_credit = range_span_credit / desired_intervals

    new_test_prices = np.arange(
        lower_bound_price_credit, upper_bound_price_credit, step_size_credit
    )
    new_test_prices = np.round(new_test_prices, precision)
    new_test_prices = list(np.unique(new_test_prices))

    price_combinations = [(price, price) for price in new_test_prices]

    # Convert to numpy array for further processing if needed
    price_combinations = np.array(price_combinations)

    # filtered_combinations = price_combinations[price_combinations[:, 1] > price_combinations[:, 0]]

    # filtered_combinations = price_combinations
    uncertainty_metrics = pd.DataFrame(
        columns=[
            "Price Credit",
            "Price Cash",
            # 'Cash Prediction',
            "Credit Prediction",
            "Actual Credit Units Sold",
            # 'Actual Cash',
            "Actual Scaled Credit Price",
            "Actual Scaled Cash Price",
            "Actual Competitor Price",
            "Cost of Sale Credit",
            # 'Total Prediction',
            # 'Cash Revenue',
            "Credit Pred Revenue",
            # 'Total Pred Revenue'
        ]
    )

    yesterdays_data = df[df["ds"] == date_bef]
    yesterdays_net_price = (
        yesterdays_data["Scaled_Price_Credit_A_day1"].values.mean()
        if not yesterdays_data.empty
        else "N/A"
    )
    yesterdays_quantity = (
        yesterdays_data[target].values.sum() if not yesterdays_data.empty else "N/A"
    )
    actual_quantity = (
        df[df["ds"] == date][target].values.sum()
        if not df[df["ds"] == date].empty
        else 0
    )
    actual_net_price = (
        df[df["ds"] == date]["Scaled_Price_Credit_A_day1"].values.mean()
        if not df[df["ds"] == date].empty
        else 0
    )
    # gmv_20_price = features[features['ds'] == date_bef]['unit_cost_mean'].mean() * 1.20 if not yesterdays_data.empty else "N/A"
    actual_quantity_credit = (
        df[(df["ds"] == date) & (df[group_by_column] == product_name)][
            target
        ].values.sum()
        if not df[df["ds"] == date].empty
        else "N/A"
    )

    actual_price_credit = (
        df[(df["ds"] == date) & (df[group_by_column] == product_name)][
            "Scaled_Price_Credit_A_day1"
        ].values.sum()
        if not df[df["ds"] == date].empty
        else "N/A"
    )
    actual_price_cash = (
        df[(df["ds"] == date) & (df[group_by_column] == product_name)][
            "Scaled_Price_Cash_A_day1"
        ].values.sum()
        if not df[df["ds"] == date].empty
        else "N/A"
    )
    actual_cost = (
        df[(df["ds"] == date) & (df[group_by_column] == product_name)][
            "Cost of Sale Credit"
        ].values.sum()
        if not df[df["ds"] == date].empty
        else "N/A"
    )

    actual_revenue = actual_net_price * actual_quantity
    actual_comp_price = None

    def get_competitive_price(features, date, product_name_int):
        row = features[
            (features["ds"] == date) & (features[group_by_column] == product_name)
        ]
        if not row.empty:
            if not pd.isna(row["Coppel Digital Price"].values[0]):
                return row["Coppel Digital Price"].values[0]
            elif not pd.isna(row["Elektra Digital Price"].values[0]):
                return row["Elektra Digital Price"].values[0]
            elif not pd.isna(row["Price To Elektra Credit (A)_day1"].values[0]):
                return row["Price To Elektra Credit (A)_day1"].values[0]
        return None

    actual_comp_price = get_competitive_price(features, date, product_name)

    for price_credit, price_cash in price_combinations:
        # price, price2 in price_combinations

        test_vector_copy = df[
            (df["ds"] == date) & (df[group_by_column] == product_name)
        ][columns_3d].drop(columns_to_drop, axis=1)

        for col, value in {
            "price_gap": test_vector_copy["price_gap"]
            - test_vector_copy["Scaled_Price_Credit_A_day1"]
            + price_credit,
            "Scaled_Price_Credit_Cash_A_day1_diff": test_vector_copy[
                "Scaled_Price_Credit_Cash_A_day1_diff"
            ]
            - test_vector_copy["Scaled_Price_Credit_A_day1"]
            + price_credit
            if "Scaled_Price_Credit_Cash_A_day1_diff" in test_vector_copy.columns
            else None,
            "Scaled_Price_Credit_Cash_A_day2_diff": test_vector_copy[
                "Scaled_Price_Credit_Cash_A_day2_diff"
            ]
            - test_vector_copy["Scaled_Price_Credit_A_day2"]
            + price_credit
            if "Scaled_Price_Credit_Cash_A_day2_diff" in test_vector_copy.columns
            else None,
            "Scaled_Price_Credit_Cash_A_day3_diff": test_vector_copy[
                "Scaled_Price_Credit_Cash_A_day3_diff"
            ]
            - test_vector_copy["Scaled_Price_Credit_A_day3"]
            + price_credit
            if "Scaled_Price_Credit_Cash_A_day3_diff" in test_vector_copy.columns
            else None,
            "Scaled_Price_Credit_A_day1": price_credit
            if "Scaled_Price_Credit_A_day1" in test_vector_copy.columns
            else None,
            "Scaled_Price_Credit_A_day2": price_credit
            if "Scaled_Price_Credit_A_day2" in test_vector_copy.columns
            else None,
            "Scaled_Price_Credit_A_day3": price_credit
            if "Scaled_Price_Credit_A_day3" in test_vector_copy.columns
            else None,
            "Scaled_Price_Credit_A_day4": price_credit
            if "Scaled_Price_Credit_A_day4" in test_vector_copy.columns
            else None,
            "Scaled_Price_Cash_A_day1": price_cash
            if "Scaled_Price_Cash_A_day1" in test_vector_copy.columns
            else None,
            "Scaled_Price_Cash_A_day2": price_cash
            if "Scaled_Price_Cash_A_day2" in test_vector_copy.columns
            else None,
            "Scaled_Price_Cash_A_day3": price_cash
            if "Scaled_Price_Cash_A_day3" in test_vector_copy.columns
            else None,
            "Scaled_Price_Cash_A_day4": price_cash
            if "Scaled_Price_Cash_A_day4" in test_vector_copy.columns
            else None,
            # "time_price_interaction": test_vector_copy["time_price_interaction"]
            # / test_vector_copy["Scaled_Price_Credit_A_day1"]
            # * price_credit
            # if "time_price_interaction" in test_vector_copy.columns
            # else None,
        }.items():
            if col in test_vector_copy.columns:
                test_vector_copy[col] = value

        if model_name == "LightGBMWithUncertainty":
            pred_val_hourly = forcaster.predict(test_vector_copy)
            lower_pred = np.sum(pred_val_hourly[0.1])
            median_pred = np.sum(pred_val_hourly[0.5])
            upper_pred = np.sum(pred_val_hourly[0.9])

        elif model_name == "CatBoostRegressorWithUncertainity":
            predictions, uncertainties = forcaster.predict(test_vector_copy)
            median_pred = np.sum(predictions)
            lower_pred = np.sum(predictions - uncertainties)
            upper_pred = np.sum(predictions + uncertainties)

        elif model_name == "CatBoostMultiOutputRegressor":
            predictions = forcaster.predict(test_vector_copy)
            predictions_cash = round(predictions[:, 1].sum())
            predictions_credit = round(predictions[:, 0].sum())

            lower_pred = predictions_cash
            median_pred = predictions_credit
            upper_pred = predictions_cash + predictions_credit

        else:
            pred_val_hourly = forcaster.predict(test_vector_copy)
            lower_pred = median_pred = upper_pred = max(
                math.ceil(sum(pred_val_hourly)), 0
            )

        lower_revenue = lower_pred * price_cash
        median_revenue = median_pred * price_credit
        upper_revenue = lower_revenue + median_revenue

        new_row = pd.DataFrame(
            [
                {
                    "Price Credit": scale_back_prices(
                        criterion_value=product_name,
                        scaled_price=np.round(price_credit, 3),
                        first_price_df=sku_map,
                        criterion_type=group_by_column,
                    ),
                    "Price Cash": scale_back_prices(
                        criterion_value=product_name,
                        scaled_price=np.round(price_cash, 3),
                        first_price_df=sku_map,
                        criterion_type=group_by_column,
                    ),
                    # 'Cash Prediction': lower_pred,
                    "Cost of Sale Credit": actual_cost,
                    "Credit Prediction": median_pred,
                    "Actual Competitor Price": actual_comp_price,
                    # 'Actual Cash': actual_quantity_cash,
                    "Actual Credit Units Sold": actual_quantity_credit,
                    "Actual Scaled Credit Price": scale_back_prices(
                        criterion_value=product_name,
                        scaled_price=np.round(actual_price_credit, 3),
                        first_price_df=sku_map,
                        criterion_type=group_by_column,
                    ),
                    "Actual Scaled Cash Price": scale_back_prices(
                        criterion_value=product_name,
                        scaled_price=np.round(actual_price_cash, 3),
                        first_price_df=sku_map,
                        criterion_type=group_by_column,
                    ),
                    # 'Total Prediction': upper_pred,
                    # 'Cash Revenue': lower_revenue,
                    "Credit Pred Revenue": median_revenue,
                    # 'Total Revenue': upper_revenue,
                }
            ]
        )

        uncertainty_metrics = pd.concat(
            [uncertainty_metrics, new_row], ignore_index=True
        )

    uncertainityMetrics_filename = f"{folder_name}/Inference_Graphs/{sub_folder_name}/{test_date}/{product_name}/{date}_uncertainty_metrics.csv"
    uncertainty_metrics.to_csv(f"{uncertainityMetrics_filename}")

    if product_name == "Pimentón Maduración Mixta Semi (Mediano / Nataly) Desde 2Kg":
        product_name = product_name.replace("/", "-")
    if product_name == "Pimentón Maduración Mixta Semi (Mediano / Nataly) Desde 5kg":
        product_name = product_name.replace("/", "-")

    plot_3d_volume(
        uncertainty_metrics, folder_name, sub_folder_name, test_date, product_name, date
    )
    plot_3d_Revenue(
        uncertainty_metrics, folder_name, sub_folder_name, test_date, product_name, date
    )
    return
