import pandas as pd


def load_data(
    customer_path="data/raw/customers.csv",
    applications_path="data/raw/credit_applications.csv",
):
    # Load the datasets
    customers_df = pd.read_csv(customer_path, index_col=0)
    applications_df = pd.read_csv(applications_path, index_col=0)

    return customers_df, applications_df


def preprocess_data(customers_df, applications_df):
    """
    Merge the customers and credit applications data and save the result as a pickle file.
    """

    # Merge the datasets
    merged_df = pd.merge(
        customers_df, applications_df, on=["client_nr", "yearmonth"], how="outer"
    )

    # Convert the yearmonth column to a datetime column
    merged_df["date"] = pd.to_datetime(merged_df["yearmonth"], format="%Y%m")
    merged_df.drop("yearmonth", axis=1, inplace=True)

    # Fill the missing values in the CRG column with 3
    merged_df["CRG"].fillna(3, inplace=True)

    int_columns = ["client_nr", "CRG"]
    merged_df[int_columns] = merged_df[int_columns].astype(int)

    return merged_df


def main():
    customers_df, applications_df = load_data()
    df = preprocess_data(customers_df, applications_df)

    # Save the merged dataset as a pickle file
    df.to_pickle("data/processed/processed.pkl")


if __name__ == "__main__":
    main()
