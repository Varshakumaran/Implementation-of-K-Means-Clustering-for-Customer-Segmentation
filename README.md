
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.1.Start by importing the required libraries (pandas, matplotlib.pyplot, KMeans from sklearn.cluster).
 2.Load the Mall_Customers.csv dataset into a DataFrame.
 3.Check for missing values in the dataset to ensure data quality.
 4.Select the features Annual Income (k$) and Spending Score (1-100) for clustering.
 5.Use the Elbow Method by running KMeans for cluster counts from 1 to 10 and record the Within
Cluster Sum of Squares (WCSS).
 6.Plot the WCSS values against the number of clusters to determine the optimal number of clusters
 (elbow point).
 7.Fit the KMeans model to the selected features using the chosen number of clusters (e.g., 5).
 8.Predict the cluster label for each data point and assign it to a new column called cluster.
 9.Split the dataset into separate clusters based on the predicted labels.
 10.Visualize the clusters using a scatter plot, and optionally mark the cluster centroids

## Program:
```

Developed by: Varsha k
RegisterNumber:  212223220122

```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Display basic information
print(data.head())
print(data.info())
print(data.isnull().sum())

# Elbow method to find the optimal number of clusters
wcss = []
print("Name: A.Lahari")
print("Reg.No: 212223230111")

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(data.iloc[:, 3:])
    wcss.append(kmeans.inertia_)

# Plot the elbow method
plt.plot(range(1, 11), wcss)
plt.xlabel("No. of clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Applying KMeans with optimal clusters (e.g., 5)
km = KMeans(n_clusters=5, init="k-means++", random_state=42)
km.fit(data.iloc[:, 3:])
y_pred = km.predict(data.iloc[:, 3:])

# Add cluster column to the dataset
data["cluster"] = y_pred

# Separate the clusters
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]

# Plotting the clusters
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c="red", label="Cluster 0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c="black", label="Cluster 1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c="blue", label="Cluster 2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c="green", label="Cluster 3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c="magenta", label="Cluster 4")

# Final plot settings
plt.legend()
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

```
## data.head()
![Screenshot 2025-05-23 180630](https://github.com/user-attachments/assets/eb8e3dbb-f744-4948-b6d3-8f27ff15cf9a)
## data.info()
![Screenshot 2025-05-23 180651](https://github.com/user-attachments/assets/fffe58a2-0e4a-4e67-a3a0-cc443fae5cd4)
## data.isnull().sum()
![Screenshot 2025-05-23 180726](https://github.com/user-attachments/assets/fcc61f0f-1eee-4789-8970-b17fe44439f1)
## ELbow method
![Screenshot 2025-05-23 180753](https://github.com/user-attachments/assets/77798174-dc5b-439d-875b-a45b9b02c521)
## KMeans
![Screenshot 2025-05-23 180818](https://github.com/user-attachments/assets/9eceeb6a-826c-4fef-85ae-b693f42de488)
## y_pred
![Screenshot 2025-05-23 180838](https://github.com/user-attachments/assets/8ccb62fe-ef87-4907-ac12-826fa3210348)
##  Customer Segments
![Screenshot 2025-05-23 180940](https://github.com/user-attachments/assets/7eb9f8d6-01d6-4b86-a79a-6741d88b3aa4)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
