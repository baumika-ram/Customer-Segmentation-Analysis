import os
import pandas as pd

print("Current Working Directory:", os.getcwd())
print("Files in current folder:", os.listdir())

df = pd.read_excel("Customer_Segmentation_Dataset.xlsx")

print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

from sklearn.preprocessing import StandardScaler

# Select features for clustering
features = df[['Income', 'Total_Spend', 'Purchase_Frequency', 'Average_Order_Value']]

print("\nSelected Features:")
print(features.head())

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print("\nScaled Data:")
print(scaled_features[:5])

from sklearn.cluster import KMeans

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model and predict clusters
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Display Customer ID and Cluster
print("\nCustomer Segments:")
print(df[['Customer_ID', 'Cluster']])

# Cluster-wise Summary
cluster_summary = df.groupby('Cluster')[['Income', 'Total_Spend',
                                         'Purchase_Frequency',
                                         'Average_Order_Value']].mean()

print("\nCluster Summary:")
print(cluster_summary)

# Rename cluster numbers to meaningful names
cluster_names = {
    0: "Regular Customers",
    1: "Low Value Customers",
    2: "High Value Customers"
}

df["Customer_Segment"] = df["Cluster"].map(cluster_names)

print("\nCustomer Segments:")
print(df[["Customer_ID", "Customer_Segment"]])

# Save the updated dataset
df.to_excel("Customer_Segmentation_Result.xlsx", index=False)

print("\nCustomer segmentation completed successfully!")
print("Updated dataset saved as Customer_Segmentation_Result.xlsx")