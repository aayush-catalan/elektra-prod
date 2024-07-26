import pandas as pd
import numpy as np
import os

national_price = pd.read_csv(
                    "data/national_price_20240722.csv", index_col=0
                )

def find_catalan_prediction(
    inference_graphs_dir="Test_30Days/Inference_graphs",
    folder_date="2024-01-22",
    product_id="US com",
    model_names=["CatBoostRegressor", "LightGBM"],
    test_date="2023-11-30",
):
    for model_name in model_names:
        base_path = os.path.join(inference_graphs_dir, model_name, folder_date)

        if not os.path.exists(base_path):
            continue

        product_dir = os.path.join(base_path, product_id)
        csv_file = os.path.join(
            product_dir, f'{test_date.strftime("%Y-%m-%d")}_uncertainty_metrics.csv'
        )

        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)

            try:
                test_date = pd.to_datetime(test_date)

                # national_price = pd.read_csv(
                #     "data/national_price_06142024.csv", index_col=0
                # )
                filtered_row = national_price[national_price["SKU"] == int(product_id)]

                lower_bound = filtered_row["Control Credit Precio A"].iloc[0] * 0.95
                upper_bound = filtered_row["Control Credit Precio A"].iloc[0] * 1.05

                lower_bound_cash = (
                    filtered_row["Control Cash Precio A"].iloc[0] * 0.90
                )
                upper_bound_cash = (
                    filtered_row["Control Cash Precio A"].iloc[0] * 1.10
                )

                filtered_df = df[
                    (df["Price Credit"] >= lower_bound)
                    & (df["Price Credit"] <= upper_bound)
                    & (df["Price Cash"] >= lower_bound_cash)
                    & (df["Price Cash"] <= upper_bound_cash)
                ]

                optimal_row = filtered_df.loc[filtered_df["Credit Prediction"].idxmax()]

            except ValueError:
                print("Could not find an upper and lower bound")
                continue

            return optimal_row

        else:
            print("no path for date")


def select_price_for_not_priced(not_priced, date):
    # not_priced = pd.read_csv("Physical_sku_not_priced_june_21.csv")
    products = not_priced["SKU"].values
    # products = np.append(products, 31058674)
    products

    # national_price = pd.read_csv("data/national_price_06142024.csv", index_col=0)
    predictions = pd.DataFrame()
    for product_id in products:
        print(product_id)
        filtered_row = national_price[national_price["SKU"] == int(product_id)]
        print("filtered_row", filtered_row)
        credit_A_price = filtered_row["Control Credit Precio A"].iloc[0] * 0.97

        Cash_A_price = filtered_row["Control Cash Precio A"].iloc[0] * 0.97

        new_row = pd.DataFrame(
            [
                {
                    "Date": date,
                    "SKU": product_id,
                    "Best Price Credit A": credit_A_price,
                    "Best Price Cash A": Cash_A_price,
                    "Best Price Credit DE": filtered_row["Control Precio DE ($)"].iloc[0],
                    "Best Price Cash DE": filtered_row["Control Precio DE ($)"].iloc[0],
                    "Credit Sales Prediction": 0,
                    "Predicted Revenue": 0,
                    "Cost of Sale Credit": filtered_row["Cost of Sale Credit"].iloc[0],
                    # 'Actual Credit Units Sold': optimal_row['Actual Credit Units Sold'],
                    # 'Actual Cash Price A': optimal_row['Actual Scaled Cash Price'],
                    # 'Actual Credit Price A': optimal_row['Actual Scaled Credit Price'],
                    # 'Actual Revenue': optimal_row['Actual Scaled Credit Price'] * optimal_row['Actual Credit Units Sold']
                }
            ]
        )

        numeric_columns = new_row.select_dtypes(include=["number"]).columns
        print(numeric_columns)
        # new_row[numeric_columns] = new_row[numeric_columns].round(0).astype(int)
        for col in numeric_columns:
            new_row[col] = new_row[col].astype(int)

        predictions = pd.concat([predictions, new_row])
    # predictions.to_csv("final_not_priced_june21.csv")
    return predictions


def generate_predictions(
    dates, physical_sku_ls, inference_graphs_dir, folder_date, model_names
):
    predictions = pd.DataFrame()
    sku_not_priced = []

    for sku in physical_sku_ls:
        for date in dates:
            try:
                optimal_row = find_catalan_prediction(
                    inference_graphs_dir=inference_graphs_dir,
                    folder_date=folder_date,
                    product_id=str(sku),
                    model_names=model_names,
                    test_date=pd.to_datetime(date),
                )
            except Exception as e:
                print(f"Error occurred: {e}")
                sku_not_priced.append(sku)
                continue  # Continue to the next iteration if an error occurs

            if optimal_row is None:
                print("No DataFrame returned from find_catalan_prediction.")
                sku_not_priced.append(sku)
                continue  # Continue to the next iteration if no DataFrame is returned
            
            filtered_row = national_price[national_price["SKU"] == int(sku)]
            # Create the new row
            new_row = pd.DataFrame(
                [
                    {
                        "Date": date,
                        "SKU": sku,
                        "Best Price Credit A": optimal_row["Price Credit"],
                        "Best Price Cash A": optimal_row["Price Cash"],
                        "Best Price Credit DE": filtered_row["Control Precio DE ($)"].iloc[0],
                        "Best Price Cash DE": filtered_row["Control Precio DE ($)"].iloc[0],
                        "Credit Sales Prediction": optimal_row["Credit Prediction"],
                        "Predicted Revenue": optimal_row["Price Credit"]
                        * optimal_row["Credit Prediction"],
                        "Cost of Sale Credit": optimal_row["Cost of Sale Credit"],
                        # 'Actual Credit Units Sold': optimal_row['Actual Credit Units Sold'],
                        # 'Actual Cash Price A': optimal_row['Actual Scaled Cash Price'],
                        # 'Actual Credit Price A': optimal_row['Actual Scaled Credit Price'],
                        # 'Actual Revenue': optimal_row['Actual Scaled Credit Price'] * optimal_row['Actual Credit Units Sold']
                    }
                ]
            )

            # Apply rounding and conversion to integers
            numeric_columns = new_row.select_dtypes(include=["number"]).columns
            for col in numeric_columns:
                new_row[col] = new_row[col].round(0).astype(int)

            # Concatenate the new row to the predictions DataFrame
            predictions = pd.concat([predictions, new_row], ignore_index=True)

    df_sku_not_priced = pd.DataFrame(sku_not_priced, columns=["SKU"])

    # Assuming select_price_for_not_priced is a defined function
    df_not_priced = select_price_for_not_priced(
        not_priced=df_sku_not_priced, date=dates[0]
    )
    df_priced = predictions

    combined = pd.concat([df_priced, df_not_priced])
    combined.to_csv(
        "data/predictions/07262024_EKT_Physical_Stores_Prices_Final_temp.csv"
    )

    return combined
