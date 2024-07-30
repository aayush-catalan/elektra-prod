import pandas as pd


def validation_check_price_recommendations(
    commercial_comments_path: str, predictions_path: str
):
    # Read the Excel file using Pandas
    commercial_comments_df = pd.read_excel(commercial_comments_path, header=2)

    # Select and rename columns
    recommendations_df = commercial_comments_df[
        [
            "sku_number",
            "Estatus",
            "Precio DE ($)",
            "Precio A ($)",
            "Precio A ($).1",
            "Costo Unitario",
            "Margen Objetivo Neto Telefonía",
        ]
    ].rename(
        columns={
            "Precio A ($)": "Control Cash Precio A",
            "Precio A ($).1": "Control Credit Precio A",
            "Precio DE ($)": "Control Precio DE ($)",
            "Costo Unitario": "Cost of Sale Credit",
            "Margen Objetivo Neto Telefonía": "Margin",
            "sku_number": "SKU",
        }
    )

    # Read the CSV file using Pandas
    final_prices = pd.read_csv(predictions_path)

    # Merge the DataFrames
    final_prices = final_prices.merge(
        recommendations_df[
            [
                "SKU",
                "Control Cash Precio A",
                "Control Precio DE ($)",
                "Cost of Sale Credit",
            ]
        ],
        on="SKU",
        how="inner",
    )

    # Find inequalities and print results
    inequal_A = final_prices[
        final_prices["Best Price Credit A"] != final_prices["Best Price Cash A"]
    ]
    print("Inequal Price A in Final Spreadsheet: " + str(len(inequal_A)))

    inequal_DE = final_prices[
        final_prices["Best Price Credit DE"] != final_prices["Best Price Cash DE"]
    ]
    print("Inequal Price DE in Final Spreadsheet: " + str(len(inequal_DE)))

    inequal_DE_national = final_prices[
        final_prices["Best Price Credit DE"] != final_prices["Control Precio DE ($)"]
    ]
    print(
        "Inequal Price DE between national and US in Final Spreadsheet: "
        + str(len(inequal_DE_national))
    )

    de_less_A = final_prices[
        (final_prices["Best Price Credit DE"] <= final_prices["Best Price Credit A"])
        | (final_prices["Best Price Cash DE"] <= final_prices["Best Price Cash A"])
    ]
    print("DE < A in Final Spreadsheet: " + str(len(de_less_A)))

    dates = final_prices["Date"].nunique()
    print("Number of distinct dates in final prices: " + str(dates))

    null_values = final_prices.isnull().sum().sum()
    print("Null values: " + str(null_values))
