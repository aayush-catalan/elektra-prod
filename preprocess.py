import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np


def is_weekend(day):
    return 1 if day in [5, 6] else 0


def add_rolling_mean_with_shift(df, column_name, window, shift):
    new_col_name = f"rolling_mean_window_{window}_lag_{shift}"
    df[new_col_name] = df[column_name].rolling(window=window).mean().shift(shift)
    return df


def join_holidays(dataframe1, dataframe2):
    dataframe1["ds"] = pd.to_datetime(dataframe1["ds"])
    dataframe2["date"] = pd.to_datetime(dataframe2["date"])
    merged_df = pd.merge(
        dataframe1, dataframe2, how="left", left_on="ds", right_on="date"
    )
    holiday_flags = [
        "is_holiday",
        "is_holiday_day_after",
        "is_holiday_two_days_after",
        "is_holiday_day_before",
        "is_holiday_two_days_before",
    ]
    for flag in holiday_flags:
        merged_df[flag] = merged_df[flag].fillna(False)

    merged_df["is_holiday_day_after"] = merged_df["is_holiday"].shift(1)
    merged_df["is_holiday_two_days_after"] = merged_df["is_holiday"].shift(2)
    merged_df["is_holiday_day_before"] = merged_df["is_holiday"].shift(-1)
    merged_df["is_holiday_two_days_before"] = merged_df["is_holiday"].shift(-2)

    merged_df["holiday_name"].fillna("None")

    merged_df.drop("date", axis=1, inplace=True)

    merged_df["is_holiday_day_after"].fillna(False, inplace=True)
    merged_df["is_holiday_two_days_after"].fillna(False, inplace=True)
    merged_df["is_holiday_day_before"].fillna(False, inplace=True)
    merged_df["is_holiday_two_days_before"].fillna(False, inplace=True)

    return merged_df.drop(
        [
            "holiday_name",
            "country_code",
            "holiday_type",
            "main_activity",
            "second_activity",
            "third_activity",
        ],
        axis=1,
    )


def create_rolling_sd(df, column, window, shift):
    df[f"{column}_rolling_sd_{window}"] = (
        df[column].rolling(window, closed="left", min_periods=window).std().shift(shift)
    )
    return df


def join_holidays(dataframe1, dataframe2):
    # dataframe2 = pd.read_csv('Mexico_holidays.csv',index_col=0)
    # dataframe1 = pd.read_csv('Transformed_Current_Mirror_Combined_Cluster2_total_sales_SKUS.csv',index_col=0)

    dataframe1["ds"] = pd.to_datetime(dataframe1["ds"])
    dataframe2["date"] = pd.to_datetime(dataframe2["date"])

    merged_df = pd.merge(
        dataframe1, dataframe2, how="left", left_on="ds", right_on="date"
    )
    merged_df["is_holiday_day_after"] = merged_df["is_holiday"].shift(1)
    merged_df["is_holiday_two_days_after"] = merged_df["is_holiday"].shift(2)
    merged_df["is_holiday_day_before"] = merged_df["is_holiday"].shift(-1)
    merged_df["is_holiday_two_days_before"] = merged_df["is_holiday"].shift(-2)

    merged_df["holiday_name"].fillna("None")

    # Create features for holidays in the next week, last week, and next 4 days
    merged_df["if_holiday_next_week"] = (
        merged_df["is_holiday"].rolling(window=8).max().shift(-7)
    )
    merged_df["if_holiday_last_week"] = (
        merged_df["is_holiday"].rolling(window=8).max().shift(7)
    )
    merged_df["if_holiday_in_next_4_days"] = (
        merged_df["is_holiday"].rolling(window=5).max().shift(-4)
    )
    merged_df["if_holiday_in_last_4_days"] = (
        merged_df["is_holiday"].rolling(window=5).max().shift(4)
    )

    merged_df.drop(["holiday_name", "date"], axis=1, inplace=True)

    merged_df["is_holiday_day_after"].fillna(False, inplace=True)
    merged_df["is_holiday"].fillna(False, inplace=True)
    merged_df["is_holiday_two_days_after"].fillna(False, inplace=True)
    merged_df["is_holiday_day_before"].fillna(False, inplace=True)
    merged_df["is_holiday_two_days_before"].fillna(False, inplace=True)
    merged_df["if_holiday_next_week"].fillna(False, inplace=True)
    merged_df["if_holiday_last_week"].fillna(False, inplace=True)
    merged_df["if_holiday_in_next_4_days"].fillna(False, inplace=True)
    merged_df["if_holiday_in_last_4_days"].fillna(False, inplace=True)

    return merged_df


def assign_lifecycle_stage(sku_df):
    sku_df = sku_df.sort_values(by="Days Since Introduction")
    sku_df["pct_change"] = sku_df["y"].pct_change().fillna(0)

    stages = []
    for i, row in sku_df.iterrows():
        if row["Days Since Introduction"] <= 30:
            stages.append("Introduction")
        elif row["pct_change"] > 0.5:
            stages.append("Growth")
        elif -0.1 <= row["pct_change"] <= 0.1:
            stages.append("Maturity")
        else:
            stages.append("Decline")

    sku_df["life_cycle_stage"] = stages
    return sku_df


def add_promotion_calender(daily_df):
    calendar_df = pd.read_csv("data/special_days_with_elektra_2022_2024.csv")
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])
    calendar_df["special_day"] = calendar_df["special_day"].fillna("Regular Day")
    calendar_df = pd.get_dummies(calendar_df, columns=["special_day"])

    daily_df = daily_df.merge(
        calendar_df, left_on="ds", right_on="date", how="left"
    ).drop(columns=["date"])

    return daily_df


# def closest_friday_before(day):
#     date = pd.Timestamp(day)
#     days_to_friday = (date.weekday() - 4) % 7
#     return date - pd.Timedelta(days=days_to_friday)


def mode(x):
    return x.mode().max() if not x.mode().empty else np.nan


def scale_prices(df, date_column, price_column, new_column_name="Scaled Price"):
    df = df.sort_values(by=date_column).reset_index(drop=True)

    first_price = df.loc[0, price_column]
    scaling_factor = 1000 / first_price

    df["Scaled_Price_Credit_A"] = df[price_column] * scaling_factor
    df["Scaled_Price_Cash_A"] = df["Cash Price To Elektra (A)"] * scaling_factor
    df["Scaled_Price_Cash_DE"] = df["Elektra Cash Price (De)"] * scaling_factor
    df["Scaled_Price_Credit_DE"] = df["Elektra Credit Price (De)"] * scaling_factor
    return df


def create_aggregation_dict(group_by_column):
    agg_dict = {
        "SKU": "first",
        "Model": "first",
        "Cash Unit Sale": "sum",
        "Cash Price To Elektra (A)": mode,
        "Elektra Cash Price (De)": mode,
        "Price To Elektra Credit (A)": mode,
        "Elektra Credit Price (De)": mode,
        "Gross margin%": mode,
        "Coppel Digital Price": mode,
        "Elektra Digital Price": mode,
        "Sale of Credit Units": "sum",
        "Cost of Sale Credit": "mean",
        "Brand": "first",
    }

    hierarchy = ["Brand", "Model", "SKU"]

    if group_by_column in hierarchy:
        idx = hierarchy.index(group_by_column)
        columns_to_remove = hierarchy[idx:]
        for col in columns_to_remove:
            if col in agg_dict:
                del agg_dict[col]

    return agg_dict


def dynamic_aggregation(df, group_by_column):
    agg_dict = create_aggregation_dict(group_by_column)
    # print('agg_dict',agg_dict)
    groupby_columns = ["ds", "Pilot/Control", group_by_column]
    aggregated = df.groupby(groupby_columns).agg(agg_dict).reset_index()

    return aggregated


def add_transformations(daily_df, test_date, product_names, group_by_column):
    try:
        daily_df["ds"] = daily_df["Day of Date"]
        daily_df["ds"] = pd.to_datetime(daily_df["ds"])

        daily_df = dynamic_aggregation(daily_df, group_by_column)
        

        daily_df["y_credit"] = daily_df["Sale of Credit Units"]
        daily_df["y_cash"] = daily_df["Cash Unit Sale"]

        daily_df = daily_df[daily_df["Pilot/Control"] == "PILOTO"]

        daily_df.sort_values(by="ds", inplace=True)
        max_date = max(daily_df["ds"].max(), pd.to_datetime(test_date))

        full_date_range = pd.date_range(
            start=daily_df["ds"].min(), end=max_date, freq="D"
        )

        daily_df = (
            daily_df.set_index("ds")
            .reindex(full_date_range)
            .rename_axis("ds")
            .reset_index()
        )

        if not daily_df.empty:
            daily_df = scale_prices(daily_df, "ds", "Price To Elektra Credit (A)")

        daily_df["ds"] = pd.to_datetime(daily_df["ds"])

        daily_df.sort_values(by="ds", inplace=True)

        daily_df.fillna({"y_credit": 0, "y_cash": 0}, inplace=True)

        daily_df.fillna(method="ffill", inplace=True)
        daily_df.fillna(method="bfill", inplace=True)

        daily_df["Month"] = daily_df["ds"].dt.month
        daily_df["Year"] = daily_df["ds"].dt.year
        daily_df["Week Number"] = daily_df["ds"].dt.isocalendar().week

        # Ensure "Days since introduction" is incremented correctly
        introduction_date = daily_df["ds"].min()

        intro_price = daily_df[daily_df["ds"] == introduction_date][
            "Price To Elektra Credit (A)"
        ].values[0]

        # print('intro_price',intro_price)

        daily_df["diff_from_intro_price"] = (
            daily_df["Price To Elektra Credit (A)"] - intro_price
        )
        # print('diff_from_intro_price',daily_df["diff_from_intro_price"])
        daily_df["Days Since Introduction"] = (
            daily_df["ds"] - introduction_date
        ).dt.days

        daily_df["y_credit_3days"] = (
            daily_df["y_credit"].rolling(window=3, min_periods=1).sum().shift(-2)
        )
        daily_df["y_credit_4days"] = (
            daily_df["y_credit"].rolling(window=4, min_periods=1).sum().shift(-3)
        )

        daily_df["y_credit_max_4days_last2weeks"] = (
            daily_df["y_credit_4days"].rolling(window=14, min_periods=1).max().shift(-3)
        )
        daily_df["y_credit_max_3days_last2weeks"] = (
            daily_df["y_credit_3days"].rolling(window=14, min_periods=1).max().shift(-3)
        )

        daily_df["y_cash_3days"] = (
            daily_df["y_cash"].rolling(window=3, min_periods=1).sum().shift(-2)
        )
        daily_df["y_cash_4days"] = (
            daily_df["y_cash"].rolling(window=4, min_periods=1).sum().shift(-3)
        )

        daily_df["sale_credit_3days_lag5"] = (
            daily_df["y_credit"].rolling(window=3, min_periods=1).sum().shift(5)
        )
        daily_df["sale_credit_4days_lag6"] = (
            daily_df["y_credit"].rolling(window=4, min_periods=1).sum().shift(6)
        )

        daily_df["sale_cash_3days_lag5"] = (
            daily_df["y_cash"].rolling(window=3, min_periods=1).sum().shift(5)
        )
        daily_df["sale_cash_4days_lag6"] = (
            daily_df["y_cash"].rolling(window=4, min_periods=1).sum().shift(6)
        )

        daily_df["ratio_sales_3days_lag5"] = (
            daily_df["sale_credit_3days_lag5"] / daily_df["sale_cash_3days_lag5"]
        )
        daily_df["ratio_sales_4days_lag6"] = (
            daily_df["sale_credit_4days_lag6"] / daily_df["sale_cash_4days_lag6"]
        )

        daily_df["discount_ratio"] = (
            daily_df["Price To Elektra Credit (A)"]
            - daily_df["Cash Price To Elektra (A)"]
        ) / daily_df["Cash Price To Elektra (A)"]

        price_columns = [
            "Price To Elektra Credit (A)",
            "Cash Price To Elektra (A)",
            "Scaled_Price_Credit_A",
            "Scaled_Price_Cash_A",
            "Scaled_Price_Cash_DE",
            "Scaled_Price_Credit_DE",
            "discount_ratio",
        ]

        

        for col in price_columns:
            daily_df[f"{col}_day1"] = daily_df[col]
            daily_df[f"{col}_day2"] = daily_df[col].shift(-1)
            daily_df[f"{col}_day3"] = daily_df[col].shift(-2)
            daily_df[f"{col}_day4"] = daily_df[col].shift(-3)
            daily_df[f"{col}_mean_4days"] = (
                daily_df[f"{col}_day1"]
                + daily_df[f"{col}_day2"]
                + daily_df[f"{col}_day3"]
                + daily_df[f"{col}_day4"]
            ) / 4
            daily_df[f"{col}_mean_3days"] = (
                daily_df[f"{col}_day1"]
                + daily_df[f"{col}_day2"]
                + daily_df[f"{col}_day3"]
            ) / 3
        
        daily_df['price_gap'] = daily_df['Scaled_Price_Credit_DE_day1'] - daily_df['Scaled_Price_Credit_A_day1']

        daily_df = add_promotion_calender(daily_df)

        daily_df = daily_df.drop(
            [
                # 'Day of Date',
                "Sale of Credit Units",
                "Price To Elektra Credit (A)",
                "Cash Price To Elektra (A)",
                "Scaled_Price_Credit_A",
                "Scaled_Price_Cash_A",
                "Scaled_Price_Cash_DE",
                "Scaled_Price_Credit_DE",
            ],
            axis=1,
        )

        daily_df["day_of_week_n"] = daily_df["ds"].dt.day_of_week
        daily_df["is_weekend"] = daily_df["day_of_week_n"].apply(is_weekend)

        daily_df["Scaled_Price_Credit_Cash_A_day1_diff"] = (
            daily_df["Scaled_Price_Credit_A_day1"]
            - daily_df["Scaled_Price_Cash_A_day1"]
        )
        daily_df["Scaled_Price_Credit_Cash_A_day2_diff"] = (
            daily_df["Scaled_Price_Credit_A_day2"]
            - daily_df["Scaled_Price_Cash_A_day2"]
        )
        daily_df["Scaled_Price_Credit_Cash_A_day3_diff"] = (
            daily_df["Scaled_Price_Credit_A_day3"]
            - daily_df["Scaled_Price_Cash_A_day3"]
        )
        daily_df["Scaled_Price_Credit_Cash_A_day4_diff"] = (
            daily_df["Scaled_Price_Credit_A_day4"]
            - daily_df["Scaled_Price_Cash_A_day4"]
        )

        daily_df["3d_credit_lag7"] = daily_df["y_credit_3days"].shift(7)
        daily_df["3d_cash_lag7"] = daily_df["y_cash_3days"].shift(7)
        daily_df["4d_credit_lag7"] = daily_df["y_credit_4days"].shift(7)
        daily_df["4d_cash_lag7"] = daily_df["y_cash_4days"].shift(7)

        daily_df["rolling_mean_3days_credit_lag7"] = (
            daily_df["y_credit_3days"].rolling(window=7).mean().shift(5)
        )
        daily_df["rolling_mean_4days_credit_lag7"] = (
            daily_df["y_credit_4days"].rolling(window=7).mean().shift(5)
        )

        daily_df["rolling_mean_3days_cash_lag7"] = (
            daily_df["y_cash_3days"].rolling(window=7).mean().shift(5)
        )
        daily_df["rolling_mean_4days_cash_lag7"] = (
            daily_df["y_cash_4days"].rolling(window=7).mean().shift(5)
        )

        daily_df["total_sales_3d_last_week_lag2"] = (
            daily_df["y_credit_3days"].rolling(window=7).mean().shift(2)
        )
        daily_df["total_sales_3d_last_week_lag2"] = daily_df[
            "total_sales_3d_last_week_lag2"
        ].fillna(0)

        daily_df["total_sales_4d_last_week_lag2"] = (
            daily_df["y_credit_4days"].rolling(window=7).mean().shift(2)
        )
        daily_df["total_sales_4d_last_week_lag2"] = daily_df[
            "total_sales_4d_last_week_lag2"
        ].fillna(0)

        daily_df["ratio_price_day1"] = (
            daily_df["Price To Elektra Credit (A)_day1"]
            / daily_df["Cash Price To Elektra (A)_day1"]
        )
        daily_df["ratio_price_day2"] = (
            daily_df["Price To Elektra Credit (A)_day2"]
            / daily_df["Cash Price To Elektra (A)_day2"]
        )
        daily_df["ratio_price_day3"] = (
            daily_df["Price To Elektra Credit (A)_day3"]
            / daily_df["Cash Price To Elektra (A)_day3"]
        )
        daily_df["ratio_price_day4"] = (
            daily_df["Price To Elektra Credit (A)_day4"]
            / daily_df["Cash Price To Elektra (A)_day4"]
        )

        # daily_df['ratio_4d'] = daily_df['Price To Elektra Credit (A)_day1']/daily_df['Cash Price To Elektra (A)_day1']

        holidays_df = pd.read_csv("data/Mexico_holidays.csv", index_col=0)
        daily_df = join_holidays(daily_df, holidays_df)

        # lifecycle_stage_mapping = {
        #     'Introduction': 1,
        #     'Growth': 2,
        #     'Maturity': 3,
        #     'Decline': 4
        # }
        # daily_df = daily_df.groupby('SKU').apply(assign_lifecycle_stage).reset_index(drop=True)

        # daily_df['life_cycle_stage_encoded'] = daily_df['life_cycle_stage'].map(lifecycle_stage_mapping)

        daily_df.fillna(method="ffill", inplace=True)  # Frontfill
        daily_df.fillna(method="bfill", inplace=True)  # Backfill

        result = seasonal_decompose(
            daily_df["y_credit_3days"], model="additive", period=7
        )
        daily_df["product_residual"] = result.resid
        daily_df["product_trend"] = result.trend
        daily_df["product_seasonal"] = result.seasonal
        # daily_df["product_residual_lag2"] = daily_df["product_residual"].shift(2)
        daily_df["product_trend_3days_lag5"] = daily_df["product_trend"].shift(5)
        daily_df["product_seasonal_3days_lag5"] = daily_df["product_seasonal"].shift(5)

        # daily_df["product_trend_lag2_feature"] = daily_df["product_trend_lag2"]*daily_df['selling_price_mean']

        result = seasonal_decompose(
            daily_df["y_credit_4days"], model="additive", period=7
        )
        daily_df["product_residual"] = result.resid
        daily_df["product_trend"] = result.trend
        daily_df["product_seasonal"] = result.seasonal
        # daily_df["product_residual_lag2"] = daily_df["product_residual"].shift(2)
        daily_df["product_trend_4days_lag5"] = daily_df["product_trend"].shift(5)
        daily_df["product_seasonal_4days_lag5"] = daily_df["product_seasonal"].shift(5)

        # daily_df['if_payday'] = False

        # def closest_friday_before(date):
        #     if date.weekday() == 4:  # If the date is already a Friday
        #         return date
        #     days_to_friday = (date.weekday() - 4) % 7
        #     return date - pd.Timedelta(days=days_to_friday)

        # Add a column to track if payday
        # daily_df['payday'] = False

        # def last_day_of_month(date):
        #     next_month = date.replace(day=28) + pd.DateOffset(days=4)
        #     return next_month - pd.DateOffset(days=next_month.day)

        # # Iterate through each row and determine if the date is a payday
        # for index, row in daily_df.iterrows():
        #     date = row['ds']
        #     day = date.day

        #     # Determine the potential payday for the 15th and the 30th
        #     if day <= 15:
        #         potential_fifteenth = pd.Timestamp(year=date.year, month=date.month, day=15)
        #         payday = closest_friday_before(potential_fifteenth)
        #     else:
        #         try:
        #             potential_thirtieth = pd.Timestamp(year=date.year, month=date.month, day=30)
        #         except ValueError:
        #             potential_thirtieth = last_day_of_month(date)
        #         payday = closest_friday_before(potential_thirtieth)

        #     # Mark the date as payday if it matches the calculated payday
        #     if date == payday:
        #         daily_df.at[index, 'payday'] = True

        # daily_df['time_price_interaction'] =

        # daily_df["time_feature"] = (
        #     daily_df["Days Since Introduction"]
        #     - daily_df["Days Since Introduction"].min()
        # ) / (
        #     daily_df["Days Since Introduction"].max()
        #     - daily_df["Days Since Introduction"].min()
        # )

        # Create an interaction term between time and price
        # daily_df["time_price_interaction"] = (
        #     daily_df["time_feature"] * daily_df["Scaled_Price_Credit_A_day1"]
        # )

        daily_df.fillna(method="ffill", inplace=True)  # Frontfill
        daily_df.fillna(method="bfill", inplace=True)  # Backfill

        return daily_df

    except ValueError as e:
        print(f"Error processing data for {product_names}: {e}")
        return pd.DataFrame()
