
# Import necessary libraries

import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

# Load the dataset
customers_df = pd.read_csv("customers.csv", index_col=0)
applications_df = pd.read_csv("credit_applications.csv", index_col=0)

# Merge dataframes
df = pd.merge(customers_df, applications_df, on=[
              'client_nr', 'yearmonth'], how='outer')

# Convert yearmonth column to datetime and drop it
df["date"] = pd.to_datetime(df["yearmonth"], format="%Y%m")
df.drop("yearmonth", axis=1, inplace=True)

# Generate a Pandas profiling report for the merged dataframe
# profile = ProfileReport(df, title="Pandas Profiling Report")
# profile.to_file("clients_report.html")


# %%
# Add new features to the dataframe
df['month'] = df['date'].dt.month
df["total_volume_trx"] = df["volume_credit_trx"] + df["volume_debit_trx"]
df["balance_change"] = df["volume_credit_trx"] - df["volume_debit_trx"]
df["balance_range"] = df["max_balance"] - df["min_balance"]
df["average_credit_trx"] = (
    df["volume_credit_trx"] / df["nr_credit_trx"]).fillna(0)
df["average_debit_trx"] = (df["volume_debit_trx"] /
                           df["nr_debit_trx"]).fillna(0)

# Impute missing values in the 'CRG' column
df["CRG"] = df["CRG"].fillna(3)

# Set a new index for the dataframe
df.set_index(["client_nr", "date"], inplace=True)
# %%

# Calculate the correlation matrix of the dataframe
corr = df.corr()

# Create a heatmap using Seaborn's heatmap() function
sns.heatmap(corr, annot=False, cmap='coolwarm', fmt='.2f',
            annot_kws={"fontsize": 12, "fontweight": "bold"})

# Set the title and axis labels for the heatmap
plt.title("Correlation Heatmap", fontsize=16, fontweight="bold")
plt.xlabel("Features", fontsize=14, fontweight="bold")
plt.ylabel("Features", fontsize=14, fontweight="bold")

# Save the heatmap to a PNG file
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")

# %%

# create a copy of the dataframe with logarithmic transformations applied to some columns
log_df = df.copy()
log_columns = ["volume_credit_trx", "volume_debit_trx", "total_volume_trx", "nr_credit_trx",
               "nr_debit_trx", "total_nr_trx", "min_balance",  "max_balance", "balance_range", "balance_change", "average_credit_trx", "average_debit_trx"]
log_df[log_columns] = np.arcsinh(log_df[log_columns])

# plot histograms of the original and transformed features
df[log_columns].hist(bins=100, edgecolor='black', figsize=(15, 15))
plt.savefig("histogram_normal.png", dpi=300, bbox_inches="tight")


log_df[log_columns].hist(bins=100, edgecolor='black', figsize=(15, 15))
plt.savefig("histogram_log.png", dpi=300, bbox_inches="tight")
# %%

# calculate the mean credit application for each month
months_mean = df[["credit_application", "month"]
                 ].groupby("month").mean().reset_index()
fig = sns.barplot(
    x="month",
    y="credit_application",
    data=months_mean)
plt.title("mean credit_application per month", fontsize=12, fontweight="bold")

# save the plot to a PNG file
plt.savefig("mean_credit_application_per_month.png",
            dpi=300, bbox_inches="tight")


# %%
# plot a histogram of the total number of credit applications for each client
df["nr_credit_applications"].groupby("client_nr").sum().hist(bins=range(32))
plt.xlabel("total number credit applications", fontsize=10)
plt.ylabel("number of clients", fontsize=10)
plt.title("total credit applications per client",
          fontsize=12, fontweight="bold")

# save the plot to a PNG file
plt.savefig("total_credit_applications.png", dpi=300, bbox_inches="tight")

# %%

# Group transactions by client
grouping = df.groupby('client_nr')
diff_list = []

# Calculate months between credit applications for each client
for group in grouping:

    client_df = group[1].reset_index(level=0).asfreq(freq='MS')

    arr = client_df["credit_application"].values
    diffs = np.diff(np.where(np.array(arr) > 0))
    if diffs.any():
        diff_list.append(diffs)

arr = np.concatenate(diff_list, axis=1)[0]


# Plot histogram of months between credit applications
plt.figsize = (20, 20)
plt.hist(arr, bins=range(0, 32), density=True, align="left", rwidth=0.7, )
plt.xlabel("Months between credit applications", fontsize=10)
plt.ylabel("density", fontsize=10)

plt.savefig("months_between_credit_applications.png",
            dpi=300, bbox_inches="tight")

# %%

# Define columns to include in statistics
cols = ['total_nr_trx', 'nr_debit_trx', 'volume_debit_trx', 'nr_credit_trx',
        'volume_credit_trx', 'min_balance', 'max_balance',
        # 'credit_application', 'nr_credit_applications',
        'total_volume_trx', 'balance_change', 'balance_range',
        "average_credit_trx", "average_debit_trx"]

# Group transactions by client and calculate statistics
client_statistics = df[cols].groupby("client_nr").agg(
    ['mean', "std", 'median', "max", 'min'])
client_statistics.columns = [
    "_".join(a) for a in client_statistics.columns.to_flat_index()]


# Add client CRG data to statistics
client_crg = df["CRG"].groupby("client_nr").max()
client_df = pd.concat([client_crg, client_statistics], axis=1).fillna(0)

# %%
# Calculate sum of credit applications for each client
cmap = df['nr_credit_applications'].groupby("client_nr").agg('sum')


# Generate UMAP embeddings using cosine distance
embedding = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=2,
    random_state=42,
    metric="cosine",
).fit_transform(client_df)


# Plot UMAP embeddings with color-coded points based on sum of credit applications

fig, ax = plt.subplots(figsize=(15, 15))

scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                     c=cmap, cmap='Spectral', alpha=0.7)

plt.colorbar(scatter)
ax.set_xlabel("embedding 1")
ax.set_ylabel("embedding 2")
ax.set_title("UMAP client embeddings with cosine distance")

plt.savefig("UMAP_client_embeddings.png", dpi=300, bbox_inches="tight")
