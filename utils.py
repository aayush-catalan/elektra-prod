import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from pandas import DatetimeIndex

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor

# ####### For engine
# from Inference_Pipeline.config.settings import (
#     Env,
#     get_db_url_from_env_settings,
#     get_settings_from_env,
# )

# from Inference_Pipeline.db.db_models import  (
#     SQLModel
# )

from sqlalchemy import Engine, create_engine
from typing import Tuple


# def symmetric_mean_absolute_percentage_error(actual, predicted):
#     actual = np.array(actual)
#     predicted = np.array(predicted)
#     return np.mean(
#             2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))) * 100


def symmetric_mean_absolute_percentage_error(A, F):
    epsilon = 1e-8
    A = np.array(A)
    F = np.array(F)
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F) + epsilon))


def mda(actual, predicted):
    """Mean Directional Accuracy"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(
        (
            np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - actual[:-1])
        ).astype(int)
    )


def mae(actual, predicted):
    """Mean Absolute Error"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(np.abs(actual - predicted))


def save_feature_importances(
    forecaster, folder_name, sub_folder_name, product_name, feature_names=None
):
    try:
        if isinstance(forecaster.model, GradientBoostingRegressor):
            feature_importances = forecaster.model.feature_importances_
            feature_names = forecaster.model.feature_names_in_
        elif isinstance(forecaster.model, LGBMRegressor):
            feature_importances = forecaster.model.feature_importances_
            feature_names = forecaster.model.feature_name_
        elif isinstance(forecaster.model, RandomForestRegressor):
            feature_importances = forecaster.model.feature_importances_
            feature_names = forecaster.model.feature_names_in_
        elif hasattr(forecaster.model, "get_feature_importance"):  # For CatBoost
            feature_importances = forecaster.model.get_feature_importance()
            feature_names = forecaster.model.feature_names_
        else:
            raise AttributeError("Feature importances not available for this model.")

        feature_importances_df = pd.DataFrame(
            {"feature": feature_names, "importance": feature_importances}
        )

        sorted_feature_importances = feature_importances_df.sort_values(
            by="importance", ascending=False
        )
        output_csv_path = (
            f"{folder_name}/Feature_Importances/"
            f"{sub_folder_name}/{product_name}.csv"
        )
        sorted_feature_importances.to_csv(output_csv_path, index=False)

    except AttributeError as e:
        print(f"Error: {e}")


def get_top_features_with_must_include(
    X, y, must_include_features, n_additional_features=5
):
    # print(X.info())

    model = LGBMRegressor(random_state=100)
    model.fit(X, y)

    feature_importances = model.feature_importances_

    feature_names = X.columns
    fi_df = pd.DataFrame({"feature": feature_names, "importance": feature_importances})

    fi_df = fi_df[~fi_df["feature"].isin(must_include_features)]

    fi_df = fi_df.sort_values(by="importance", ascending=False).reset_index(drop=True)

    top_additional_features = fi_df.head(n_additional_features)["feature"].tolist()

    final_features = must_include_features + top_additional_features

    return final_features


def plot_preds(
    dates: DatetimeIndex,
    filtered_df: pd.DataFrame,
    Preds: list,
    title: str,
    end_val: int,
    folder: str,
    parent_folder_name: str,
    model_name: str,
    actual: list,
    actual_credit_price_ls: list,
    actual_credit_cost_ls: list,
    group_by_column: str,
):
    pds = Preds
    num_days = len(pds)

    smape = symmetric_mean_absolute_percentage_error(actual, pds)
    mda_ans = mda(actual, pds)
    mae_ans = mae(actual, pds)

    # Color for each weekday (Monday=0, Sunday=6)
    colors = ["red", "blue", "orange", "indigo", "yellow", "green", "violet"]
    weekday_colors = dates.dayofweek.map(lambda x: colors[x])

    plt.figure()
    plt.plot(dates, pds, color="#9747FF", marker="o", markersize=6, label="Prediction")

    # Overlay the scatter plot for colored points
    # plt.scatter(dates, pds, c=weekday_colors, edgecolor='k',s = 50)

    # Plot the actual line
    plt.plot(dates, actual, "#44546a", marker="x", markersize=6, label="Actual")

    legend = plt.legend(loc="upper right")
    legend.get_texts()[0].set_text("Catalan")
    legend.get_texts()[1].set_text("Elektra")

    result = pd.DataFrame()
    result["Date"] = dates
    result[group_by_column] = [title] * len(result)
    result["Prediction Volume"] = pds
    result["Actual Volume"] = actual
    result["Actual Credit Price"] = actual_credit_price_ls
    result["Actual Cost"] = actual_credit_cost_ls
    result.to_csv(
        f"{parent_folder_name}/Predictions/{folder}/{title}_{num_days}_days_{folder}.csv"
    )

    plt.gcf().set_size_inches(15, 5)
    name = title + f" MDA = {mda_ans} , SMAPE = {smape}, MAE= {mae_ans}"
    plt.title(name)
    plt.savefig(f"{parent_folder_name}/Graphs/{folder}/{title}_{num_days}_days.png")
    plt.close()

    metrics = pd.DataFrame()
    metrics["MDA"] = [mda_ans]
    metrics["SMAPE"] = [smape]
    metrics.to_csv(f"{parent_folder_name}/Metrics/{folder}/{title}_{num_days}_days.csv")

    return


def plot_uncertainity_revenue_graph(
    uncertainty_metrics,
    yesterdays_net_price,
    product_name,
    date,
    actual_revenue,
    actual_net_price,
    test_date,
    gmv_20_price,
    actual_comp_price,
    folder_name,
    sub_folder_name,
    actual_cost,
):
    fig_revenue = go.Figure()

    fig_revenue.add_trace(
        go.Scatter(
            x=uncertainty_metrics["Price"],
            y=uncertainty_metrics["Lower Revenue"],
            mode="lines",
            name="Lower Revenue Bound",
            line=dict(width=0),
            fill=None,
            showlegend=False,
        )
    )

    fig_revenue.add_trace(
        go.Scatter(
            x=uncertainty_metrics["Price"],
            y=uncertainty_metrics["Median Revenue"],
            mode="lines",
            name="Median Revenue",
            line=dict(color="#9d52ff"),
            fill="tonexty",
            fillcolor="rgba(68, 68, 68, 0.2)",
        )
    )

    fig_revenue.add_trace(
        go.Scatter(
            x=uncertainty_metrics["Price"],
            y=uncertainty_metrics["Upper Revenue"],
            mode="lines",
            name="Upper Revenue Bound",
            fill="tonexty",
            fillcolor="rgba(68, 68, 68, 0.2)",
            line=dict(width=0),
            showlegend=False,
        )
    )

    fig_revenue.update_layout(
        # title='Revenues with Uncertainty Bounds',
        xaxis_title="Prices",
        yaxis_title="Revenues",
        # template='plotly_white'
    )

    fig_revenue.update_layout(
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=0.2,
                y=1.08,  # y > 1 will place it above the plotting area
                xanchor="center",
                yanchor="bottom",
                text=f"Revenue Graph for {product_name} - {date}",
                font=dict(family="Arial", size=14, color="black"),
                showarrow=False,
                bgcolor="white",
                # bordercolor='green',
                # borderwidth=1
            ),
            dict(
                xref="paper",
                yref="paper",
                x=0.6,
                y=1.05,  # y > 1 will place it above the plotting area
                xanchor="center",
                yanchor="bottom",
                text=f"Yesterday's net price: {yesterdays_net_price}",
                font=dict(family="Arial", size=14, color="green"),
                showarrow=False,
                bgcolor="white",
                bordercolor="green",
                borderwidth=1,
            ),
            dict(
                xref="paper",
                yref="paper",
                x=0.2,
                y=1.05,  # Increase y to stack annotations above each other
                xanchor="center",
                yanchor="bottom",
                text=f"Actual Cost: {actual_cost}",
                font=dict(family="Arial", size=14, color="orange"),
                showarrow=False,
                bgcolor="white",
                bordercolor="orange",
                borderwidth=1,
            ),
            dict(
                xref="paper",
                yref="paper",
                x=0.8,
                y=1.05,  # Increase y to stack annotations above each other
                xanchor="center",
                yanchor="bottom",
                text=f"Actual Revenue: {actual_revenue}",
                font=dict(family="Arial", size=14, color="purple"),
                showarrow=False,
                bgcolor="white",
                bordercolor="purple",
                borderwidth=1,
            ),
            dict(
                xref="paper",
                yref="paper",
                x=1.04,
                y=1.05,  # Increase y to stack annotations above each other
                xanchor="center",
                yanchor="bottom",
                text=f"Actual Net Price: {actual_net_price}",
                font=dict(family="Arial", size=14, color="purple"),
                showarrow=False,
                bgcolor="white",
                bordercolor="purple",
                borderwidth=1,
            ),
        ],
    )

    fig_revenue.add_trace(
        go.Scatter(
            x=[
                min(uncertainty_metrics["Price"]),
                max(uncertainty_metrics["Price"]),
            ],  # This spans the whole range of prices
            y=[
                actual_revenue,
                actual_revenue,
            ],  # This keeps the line at the level of the actual quantity
            mode="lines",
            name="Actual Revenue",
            line=dict(color="red"),
        )
    )

    fig_revenue.add_trace(
        go.Scatter(
            x=[actual_net_price, actual_net_price],  # This is a vertical line
            y=[
                0,
                max(uncertainty_metrics["Upper Revenue"]),
            ],  # This spans the whole range of the y-axis
            mode="lines",
            name="Original Price",
            line=dict(color="blue", width=1),
            showlegend=True,
        )
    )

    fig_revenue.add_trace(
        go.Scatter(
            x=[10.48, 10.48],  # This is a vertical line
            y=[
                0,
                max(uncertainty_metrics["Upper Revenue"]),
            ],  # This spans the whole range of the y-axis
            mode="lines",
            name="Catalan Suggested Price",
            line=dict(color="green", dash="dash"),
        )
    )

    revenue_filename = f"{folder_name}/Inference_Graphs/{sub_folder_name}/{test_date}/{product_name}/{date}_Revenue.html"
    fig_revenue.write_html(revenue_filename)


def plot_uncertainity_demand_graph(
    uncertainty_metrics,
    yesterdays_net_price,
    product_name,
    date,
    actual_quantity,
    actual_net_price,
    test_date,
    gmv_20_price,
    yesterdays_quantity,
    actual_comp_price,
    folder_name,
    sub_folder_name,
    actual_cost,
):
    fig_volume = go.Figure()

    fig_volume.update_layout(
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=0.2,
                y=1.12,  # y > 1 will place it above the plotting area
                xanchor="center",
                yanchor="bottom",
                text=f"Demand Graph for {product_name} - {date}",
                font=dict(family="Arial", size=14, color="black"),
                showarrow=False,
                bgcolor="white",
            ),
            dict(
                xref="paper",
                yref="paper",
                x=0.6,
                y=1.05,  # y > 1 will place it above the plotting area
                xanchor="center",
                yanchor="bottom",
                text=f"Yesterday's net price: {yesterdays_net_price}",
                font=dict(family="Arial", size=14, color="green"),
                showarrow=False,
                bgcolor="white",
                bordercolor="green",
                borderwidth=1,
            ),
            dict(
                xref="paper",
                yref="paper",
                x=0.4,
                y=1.05,  # Increase y to stack annotations above each other
                xanchor="center",
                yanchor="bottom",
                text=f"Yesterday's quantity: {yesterdays_quantity}",
                font=dict(family="Arial", size=14, color="blue"),
                showarrow=False,
                bgcolor="white",
                bordercolor="blue",
                borderwidth=1,
            ),
            dict(
                xref="paper",
                yref="paper",
                x=0.2,
                y=1.05,  # Increase y to stack annotations above each other
                xanchor="center",
                yanchor="bottom",
                text=f"Actual cost: {actual_cost}",
                font=dict(family="Arial", size=14, color="orange"),
                showarrow=False,
                bgcolor="white",
                bordercolor="orange",
                borderwidth=1,
            ),
            dict(
                xref="paper",
                yref="paper",
                x=0.8,
                y=1.05,  # Increase y to stack annotations above each other
                xanchor="center",
                yanchor="bottom",
                text=f"Actual Quantity: {actual_quantity}",
                font=dict(family="Arial", size=14, color="purple"),
                showarrow=False,
                bgcolor="white",
                bordercolor="purple",
                borderwidth=1,
            ),
            dict(
                xref="paper",
                yref="paper",
                x=1.04,
                y=1.05,  # Increase y to stack annotations above each other
                xanchor="center",
                yanchor="bottom",
                text=f"Actual Net Price: {actual_net_price}",
                font=dict(family="Arial", size=14, color="purple"),
                showarrow=False,
                bgcolor="white",
                bordercolor="purple",
                borderwidth=1,
            ),
        ],
    )

    fig_volume.update_layout(
        # title='Revenues with Uncertainty Bounds',
        xaxis_title="Prices",
        yaxis_title="Demand",
        # template='plotly_white'
    )

    fig_volume.update_layout(
        margin=dict(
            t=120
        )  # Increase top margin to ensure annotations are visible and not overlapping
    )

    fig_volume.add_trace(
        go.Scatter(
            x=uncertainty_metrics["Price"],
            y=uncertainty_metrics["Lower Prediction"],
            mode="lines",
            name="Lower Bound",
            line=dict(width=0),
            fill=None,
            showlegend=True,
        )
    )

    fig_volume.add_trace(
        go.Scatter(
            x=uncertainty_metrics["Price"],
            y=uncertainty_metrics["Median Prediction"],
            mode="lines",
            name="Median Prediction",
            line=dict(color="#9d52ff"),
            fill="tonexty",
            fillcolor="rgba(68, 68, 68, 0.1)",
        )
    )

    fig_volume.add_trace(
        go.Scatter(
            x=uncertainty_metrics["Price"],
            y=uncertainty_metrics["Upper Prediction"],
            mode="lines",
            name="Upper Bound",
            fill="tonexty",
            fillcolor="rgba(68, 68, 68, 0.3)",
            line=dict(width=0),
            showlegend=False,
        )
    )

    # fig_volume.add_trace(go.Scatter(
    #     x=[actual_comp_price, actual_comp_price],  # This is a vertical line
    #     y=[0, max(uncertainty_metrics['Upper Prediction'])],  # This spans the whole range of the y-axis
    #     mode='lines',
    #     name='Actual Competitor Price',
    #     line=dict(color='red')
    # ))

    fig_volume.add_trace(
        go.Scatter(
            x=[actual_net_price, actual_net_price],  # This is a vertical line
            y=[
                0,
                max(uncertainty_metrics["Upper Prediction"]),
            ],  # This spans the whole range of the y-axis
            mode="lines",
            name="Original Price",
            line=dict(color="blue", width=1),
            showlegend=True,
        )
    )

    # fig_volume.add_trace(go.Scatter(
    #     x=[actual_cost, actual_cost],  # This is a vertical line
    #     y=[0, max(uncertainty_metrics['Upper Prediction'])],  # This spans the whole range of the y-axis
    #     mode='lines',
    #     name='Actual Cost',
    #     line=dict(color='blue', width=1),
    #     showlegend=True
    # ))

    fig_volume.update_xaxes(
        range=[min(uncertainty_metrics["Price"]), max(uncertainty_metrics["Price"])]
    )

    fig_volume.update_yaxes(autorange=True)
    volume_filename = f"{folder_name}/Inference_Graphs/{sub_folder_name}/{test_date}/{product_name}/{date}_Volume.html"
    fig_volume.write_html(volume_filename)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_volume(df, folder_name, sub_folder_name, test_date, product_name, date):
    # df = pd.read_csv('Experiments/Added Promotions/Inference_Graphs/CatBoostRegressor/2024-03-01/31051762/2024-03-01_uncertainty_metrics.csv')

    min_price = min(df["Price Credit"].min(), df["Price Cash"].min())
    max_price = max(df["Price Credit"].max(), df["Price Cash"].max())

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        df["Price Credit"], df["Price Cash"], df["Credit Prediction"], c="b", marker="o"
    )

    actual_vol = df["Actual Credit Units Sold"].iloc[0]

    ax.set_title(f"3Dplot for Date - {date} volume - Actual Volume - {actual_vol}")
    ax.set_xlabel("Price Credit")
    ax.set_ylabel("Price Cash")
    ax.set_zlabel("Credit Prediction")

    ax.set_xlim([min_price, max_price])
    ax.set_ylim([min_price, max_price])
    ax.set_zlim([df["Credit Prediction"].min(), df["Credit Prediction"].max()])

    ax.set_box_aspect([1, 1, 0.5])

    # plt.show()
    vol_filename = f"{folder_name}/Inference_Graphs/{sub_folder_name}/{test_date}/{product_name}/{date}_Volume.png"

    plt.savefig(vol_filename)


def plot_3d_Revenue(df, folder_name, sub_folder_name, test_date, product_name, date):
    min_price = min(df["Price Credit"].min(), df["Price Cash"].min())
    max_price = max(df["Price Credit"].max(), df["Price Cash"].max())

    actual_rev = (
        df["Actual Credit Units Sold"].iloc[0]
        * df["Actual Scaled Credit Price"].iloc[0]
    )

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        df["Price Credit"],
        df["Price Cash"],
        df["Credit Pred Revenue"],
        c="b",
        marker="o",
    )

    # xx, yy = np.meshgrid(np.linspace(min_price, max_price, 10), np.linspace(min_price, max_price, 10))
    # zz = np.full(xx.shape, actual_rev)

    # ax.plot_surface(xx, yy, zz, color='r', alpha=0.5)

    ax.set_title(
        f"3D Scatter Plot of Revenue for Date - {date} - Actual Rev = {actual_rev}"
    )
    ax.set_xlabel("Price Credit")
    ax.set_ylabel("Price Cash")
    ax.set_zlabel("Credit Pred Revenue")

    ax.set_box_aspect([1, 1, 0.5])

    # plt.show()

    rev_filename = f"{folder_name}/Inference_Graphs/{sub_folder_name}/{test_date}/{product_name}/{date}_revenue.png"
    plt.savefig(rev_filename)


# def get_engine(env: Env) -> Tuple[Engine, str]:
#     env_settings = get_settings_from_env(env)
#     url = get_db_url_from_env_settings(env_settings)
#     engine = create_engine(url)

#     SQLModel.metadata.create_all(engine)
#     return engine, url


def scale_prices(
    group, date_column="Day of Date", price_column="Price To Elektra Credit (A)"
):
    group = group.sort_values(by=date_column).reset_index(drop=True)

    if len(group) == 0 or group[price_column].iloc[0] == 0:
        return group

    first_price = group.loc[0, price_column]
    scaling_factor = 1000 / first_price if first_price != 0 else 0

    group["Scaled_Price_Credit_A"] = group[price_column] * scaling_factor
    group["Scaled_Price_Cash_A"] = group["Cash Price To Elektra (A)"] * scaling_factor
    group["Scaled_Price_Cash_DE"] = group["Elektra Cash Price (De)"] * scaling_factor
    group["Scaled_Price_Credit_DE"] = (
        group["Elektra Credit Price (De)"] * scaling_factor
    )

    return group


def apply_scaling(df):
    # Apply the scaling function to each SKU group
    df_scaled = df.groupby("SKU").apply(scale_prices)
    return df_scaled.drop(["SKU"], axis=1).reset_index()


def update_days_since_introduction(df):
    def calculate_days_since_intro(group):
        introduction_date = group["Day of Date"].min()
        intro_price = group.loc[
            group["Day of Date"] == introduction_date, "Price To Elektra Credit (A)"
        ].values[0]
        group["diff_from_intro_price"] = (
            group["Price To Elektra Credit (A)"] - intro_price
        )
        group["Days Since Introduction"] = (
            group["Day of Date"] - introduction_date
        ).dt.days
        return group

    # Group by SKU and apply the calculation
    df = df.groupby("SKU").apply(calculate_days_since_intro)
    return df.drop(["SKU"], axis=1).reset_index()


############ How to use
# daily_data  = pd.read_csv('data/scaled_Sanitized_Master_Physical_stores_26062024.csv')
# sku_map = pd.read_csv('data/sku_product_name_map.csv',index_col=0)

# df = pd.merge(daily_data,sku_map, on='SKU',how='left')

# df_scaled = apply_scaling(df)
# df_scaled['Day of Date'] = pd.to_datetime(df_scaled['Day of Date'])
# df_scaled = df_scaled.drop(['level_1'],axis=1)

# df_updated = update_days_since_introduction(df_scaled)
# df_updated = df_updated.drop(['level_1'],axis=1)
