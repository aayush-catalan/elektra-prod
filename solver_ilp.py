# config.py
import os

import pandas as pd
import xarray as xr
import numpy as np
import cpmpy as cp
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initial_setup(config: Dict) -> Dict[str, xr.DataArray]:
    try:
        inference_graphs_dir = f"{config['FOLDER_NAME']}/Inference_Graphs"
        all_product_paths = []

        for model_name in config["MODEL_NAMES"]:
            base_path = os.path.join(inference_graphs_dir, model_name, config["DATE"])
            if not os.path.exists(base_path):
                logger.warning(f"Path does not exist: {base_path}")
                continue

            all_product_paths.extend(
                [
                    (product_id, os.path.join(base_path, product_id))
                    for product_id in os.listdir(base_path)
                    if product_id.isdigit()
                ]
            )

        sorted_product_paths = sorted(all_product_paths)

        sku_map = pd.read_csv(config["SKU_MAP_PATH"])
        national_price_df = pd.read_csv(config["NATIONAL_PRICE_PATH"], index_col=0)

        data = {
            "prices": [],
            "revenue": [],
            "cost": [],
            "demand": [],
            "product_ids": [],
            "national_price_creditA": [],
            "national_price_cashA": [],
            "national_price_DE": [],
            "margin": [],
            "min_credit_A": [],
            "max_credit_A": [],
            "price_segment_cluster": [],
        }

        for product_id, product_dir in sorted_product_paths:
            csv_file = os.path.join(
                product_dir, f'{config["DATE"]}_uncertainty_metrics.csv'
            )
            if not os.path.exists(csv_file):
                continue

            df = pd.read_csv(csv_file)[: config["SIZE"]]
            df = df.astype(
                {
                    "Price Credit": int,
                    "Price Cash": int,
                    "Credit Prediction": int,
                    "Credit Pred Revenue": int,
                }
            )

            data["prices"].append(
                [list(pair) for pair in zip(df["Price Credit"], df["Price Cash"])]
            )
            data["revenue"].append(df["Credit Pred Revenue"].tolist())
            data["demand"].append(df["Credit Prediction"].tolist())
            data["cost"].append(abs(df["Cost of Sale Credit"]).astype(int).tolist())
            data["product_ids"].append(product_id)

            product_national_price = national_price_df[
                national_price_df["SKU"] == int(product_id)
            ].iloc[0]

            data["national_price_creditA"].append(
                int(product_national_price["Control Credit Precio A"])
            )
            data["national_price_cashA"].append(
                int(product_national_price["Control Cash Precio A"])
            )
            data["national_price_DE"].append(
                int(product_national_price["Control Precio DE ($)"])
            )
            data["min_credit_A"].append(int(df["Price Credit"].min()))
            data["max_credit_A"].append(int(df["Price Credit"].max()))
            data["price_segment_cluster"].append(
                product_national_price["price_segment"]
            )
            data["margin"].append(
                int(
                    (df["Price Credit"].mean() - df["Cost of Sale Credit"].mean())
                    / df["Price Credit"].mean()
                )
            )

        price_pair_options = list(range(1, 101))

        xr_data = {
            "prices_xr": xr.DataArray(
                data["prices"],
                dims=["product", "price_pair_option", "price"],
                coords={
                    "cluster": ("product", data["price_segment_cluster"]),
                    "product": data["product_ids"],
                    "price_pair_option": price_pair_options,
                },
                name="Prices",
            ),
            "demands_xr": xr.DataArray(
                data["demand"],
                dims=["product", "price_pair_option"],
                coords={
                    "product": data["product_ids"],
                    "price_pair_option": price_pair_options,
                },
                name="Demands",
            ),
            "costs_xr": xr.DataArray(
                data["cost"],
                dims=["product", "price_pair_option"],
                coords={"product": data["product_ids"]},
                name="Costs",
            ),
            "revenues_xr": xr.DataArray(
                data["revenue"],
                dims=["product", "price_pair_option"],
                coords={
                    "product": data["product_ids"],
                    "price_pair_option": price_pair_options,
                },
                name="Revenues",
            ),
            "national_price_creditA_xr": xr.DataArray(
                data["national_price_creditA"],
                dims=["product"],
                coords={"product": data["product_ids"]},
                name="NationalPriceCreditA",
            ),
            "national_price_cashA_xr": xr.DataArray(
                data["national_price_cashA"],
                dims=["product"],
                coords={"product": data["product_ids"]},
                name="NationalPriceCashA",
            ),
            "national_price_DE_xr": xr.DataArray(
                data["national_price_DE"],
                dims=["product"],
                coords={"product": data["product_ids"]},
                name="NationalPriceDE",
            ),
            "min_price_creditA_xr": xr.DataArray(
                data["min_credit_A"],
                dims=["product"],
                coords={"product": data["product_ids"]},
                name="MinPriceCreditA",
            ),
            "max_price_creditA_xr": xr.DataArray(
                data["max_credit_A"],
                dims=["product"],
                coords={"product": data["product_ids"]},
                name="MaxPriceCreditA",
            ),
            "margin_xr": xr.DataArray(
                data["margin"],
                dims=["product"],
                coords={"product": data["product_ids"]},
                name="Margins",
            ),
        }

        return xr_data
    except Exception as e:
        logger.error(f"Error in initial_setup: {str(e)}")
        raise


def define_model(xr_data: Dict[str, xr.DataArray]) -> Tuple[cp.Model, cp.IntVar]:
    try:
        num_products, num_price_pairs = (
            xr_data["prices_xr"].shape[0],
            xr_data["prices_xr"].shape[1],
        )
        price_indices = cp.intvar(
            0, num_price_pairs - 1, shape=(num_products,), name="price_indices"
        )
        model = cp.Model()

        add_revenue_objective(model, xr_data["revenues_xr"], price_indices)
        add_constraints(model, xr_data, price_indices)

        return model, price_indices
    except Exception as e:
        logger.error(f"Error in define_model: {str(e)}")
        raise


def add_revenue_objective(
    model: cp.Model, revenues_xr: xr.DataArray, price_indices: cp.IntVar
) -> None:
    total_revenue = cp.sum(
        [
            cp.Element(revenues_xr.values[p, :], price_indices[p])
            for p in range(revenues_xr.shape[0])
        ]
    )
    model.maximize(total_revenue)


def add_constraints(
    model: cp.Model, xr_data: Dict[str, xr.DataArray], price_indices: cp.IntVar
) -> None:
    for product in xr_data["prices_xr"].product.values:
        product_idx = xr_data["prices_xr"].get_index("product").get_loc(product)
        national_price_creditA = (
            xr_data["national_price_creditA_xr"].sel(product=product).values
        )
        national_price_DE = xr_data["national_price_DE_xr"].sel(product=product).values

        model += (
            cp.Element(
                xr_data["prices_xr"].sel(product=product, price=0).values,
                price_indices[product_idx],
            )
            * 100
            < national_price_DE * 100
        )

    price_segments = xr_data["prices_xr"].coords["cluster"].values
    unique_segments = np.unique(price_segments)

    for segment in unique_segments:
        segment_products = xr_data["prices_xr"].where(
            xr_data["prices_xr"].coords["cluster"] == segment, drop=True
        )
        segment_margins = xr_data["margin_xr"].where(
            xr_data["prices_xr"].coords["cluster"] == segment, drop=True
        )

        sorted_indices = np.argsort(-segment_margins.values)
        sorted_products = segment_products.isel(product=sorted_indices)

        if segment != 10:
            top_30_percent_index = int(len(sorted_products.product) * 0.3)
            top_30_products = sorted_products.isel(
                product=slice(0, top_30_percent_index)
            )
            remaining_70_products = sorted_products.isel(
                product=slice(top_30_percent_index, None)
            )

            add_price_constraints(
                model, xr_data, price_indices, top_30_products, 90, 100
            )
            add_price_constraints(
                model, xr_data, price_indices, remaining_70_products, 97, 102
            )
        else:
            add_price_constraints(
                model, xr_data, price_indices, sorted_products, 90, 100
            )


def add_price_constraints(
    model: cp.Model,
    xr_data: Dict[str, xr.DataArray],
    price_indices: cp.IntVar,
    products: xr.DataArray,
    lower_percent: int,
    upper_percent: int,
) -> None:
    for product in products.product.values:
        product_idx = xr_data["prices_xr"].get_index("product").get_loc(product)
        national_price_creditA = (
            xr_data["national_price_creditA_xr"].sel(product=product).values
        )
        price_element = cp.Element(
            xr_data["prices_xr"].sel(product=product, price=0).values,
            price_indices[product_idx],
        )

        model += price_element * 100 <= national_price_creditA * upper_percent
        model += price_element * 100 >= national_price_creditA * lower_percent


def solve_model(
    model: cp.Model,
    price_indices: cp.IntVar,
    xr_data: Dict[str, xr.DataArray],
    config: Dict,
) -> pd.DataFrame:
    try:
        if model.solve():
            solution = pd.DataFrame(
                columns=[
                    "Product",
                    "Catalan Credit A",
                    "Catalan Cash A",
                    "National Credit A",
                    "National Cash A",
                    "National DE",
                    "Min Credit A",
                    "Max Credit A",
                    "Demand",
                    "Revenue",
                ]
            )

            logger.info(f"Optimal total Revenue found: {model.objective_value()}")
            for p in range(xr_data["prices_xr"].shape[0]):
                price_index = price_indices[p].value()
                best_price_pair = xr_data["prices_xr"][p, price_index, :].values
                credit_A = xr_data["prices_xr"][p, price_index, 0].values
                cash_A = xr_data["prices_xr"][p, price_index, 1].values
                revenue = xr_data["revenues_xr"][p, price_index].values
                demand = xr_data["demands_xr"][p, price_index].values
                product = xr_data["prices_xr"].coords["product"].values[p]
                national_credit_price = (
                    xr_data["national_price_creditA_xr"].sel(product=product).values
                )
                national_cash_price = (
                    xr_data["national_price_cashA_xr"].sel(product=product).values
                )
                min_credit_price = (
                    xr_data["min_price_creditA_xr"].sel(product=product).values
                )
                max_credit_price = (
                    xr_data["max_price_creditA_xr"].sel(product=product).values
                )
                national_DE_price = (
                    xr_data["national_price_DE_xr"].sel(product=product).values
                )
                logger.info(
                    f" Product {product}: Best Price Pair Index {price_index}, Prices: {best_price_pair}, Revenue: {revenue}, Demand: {demand}"
                )

                new_row = pd.DataFrame(
                    [
                        [
                            product,
                            credit_A,
                            cash_A,
                            national_credit_price,
                            national_cash_price,
                            national_DE_price,
                            min_credit_price,
                            max_credit_price,
                            demand,
                            revenue,
                        ]
                    ],
                    columns=[
                        "Product",
                        "Catalan Credit A",
                        "Catalan Cash A",
                        "National Credit A",
                        "National Cash A",
                        "National DE",
                        "Min Credit A",
                        "Max Credit A",
                        "Demand",
                        "Revenue",
                    ],
                )
                solution = pd.concat([solution, new_row], ignore_index=True)

            solution.to_csv(config["OUTPUT_PATH"])

            return solution
        else:
            logger.warning("No solution found. Performing infeasibility analysis...")
            return None
    except Exception as e:
        logger.error(f"Error in solve_model: {str(e)}")
        raise


def select_price_for_not_priced(not_priced, date, national_price):
    products = not_priced["SKU"].values
    predictions = pd.DataFrame()
    for product_id in products:
        logger.info(f"Processing product: {product_id}")
        filtered_row = national_price[national_price["SKU"] == int(product_id)]
        logger.debug(f"Filtered row: {filtered_row}")
        credit_A_price = filtered_row["Control Credit Precio A"].iloc[0] * 0.97
        Cash_A_price = filtered_row["Control Cash Precio A"].iloc[0] * 0.97

        new_row = pd.DataFrame(
            [
                {
                    "Date": date,
                    "SKU": product_id,
                    "Best Price Credit A": credit_A_price,
                    "Best Price Cash A": Cash_A_price,
                    "Best Price Credit DE": filtered_row["Control Precio DE ($)"].iloc[
                        0
                    ],
                    "Best Price Cash DE": filtered_row["Control Precio DE ($)"].iloc[0],
                }
            ]
        )

        numeric_columns = new_row.select_dtypes(include=["number"]).columns
        logger.debug(f"Numeric columns: {numeric_columns}")
        for col in numeric_columns:
            new_row[col] = new_row[col].astype(int)

        predictions = pd.concat([predictions, new_row])
    return predictions


def post_process_results(config):
    df = pd.read_csv(config["OUTPUT_PATH"], index_col=0)
    df["SKU"] = df["Product"]
    df["Best Price Credit A"] = df["Catalan Credit A"]
    df["Best Price Cash A"] = df["Catalan Cash A"]
    df["Best Price Credit DE"] = df["National DE"]
    df["Best Price Cash DE"] = df["National DE"]
    df["Date"] = [config["DATE"]] * len(df)
    df = df.drop(
        [
            "Product",
            "Catalan Credit A",
            "Catalan Cash A",
            "National DE",
            "Min Credit A",
            "Max Credit A",
            "Demand",
            "Revenue",
            "National Credit A",
            "National Cash A",
        ],
        axis=1,
    )

    national_price = pd.read_csv(config["NATIONAL_PRICE_PATH"])

    temp = national_price.merge(df, on="SKU", how="outer", indicator=True)
    only_in_df1 = temp[temp["_merge"] == "left_only"]

    logger.info(
        "Selecting prices for products where the solver was unable to find a solution."
    )

    not_priced_solutions = select_price_for_not_priced(
        only_in_df1, config["DATE"], national_price
    )

    combined = pd.concat([df, not_priced_solutions])
    combined.to_csv(config["FINAL_OUTPUT_PATH"])
    logger.info(f"Final results saved to {config['FINAL_OUTPUT_PATH']}")


def main(CONFIG: Dict):
    try:
        xr_data = initial_setup(CONFIG)
        model, price_indices = define_model(xr_data)
        solution_df = solve_model(model, price_indices, xr_data, CONFIG)

        if solution_df is not None:
            logger.info("Solution found and saved successfully.")
            post_process_results(CONFIG)
        else:
            logger.warning("No solution found.")
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {str(e)}")


if __name__ == "__main__":
    CONFIG = {
        "FOLDER_NAME": "Experiments/Experiment 1",
        "MODEL_NAMES": ["CatBoostRegressor"],
        "DATE": "2024-07-26",
        "SIZE": 100,
        "SKU_MAP_PATH": "data/sku_product_name_map.csv",
        "NATIONAL_PRICE_PATH": "s3://elektra-data/commercial_comments/national_price_20240722.csv",
        "OUTPUT_PATH": "Solution4.csv",
        "FINAL_OUTPUT_PATH": "data/predictions/07262024_EKT_Physical_Stores_Prices_Test.csv",
    }
    main(CONFIG)
