import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


print("Python EDA started!")


#load the dataset
df = pd.read_csv("ecommerce_sales_analytics_5000.csv")

print("\nDataset loaded successfully!\n")
print(df.head())

#basic information about the dataset
print("\n---Dataset Shape---")
print(df.shape)

print("\n ---column names---")
print(df.columns)

print("\n---Data types---")
print(df.dtypes)

print("\n---Missing values---")
print(df.isnull().sum())

print("\n---duplicate rows---")
print(df.duplicated().sum())

print("\n---Statistical summary---")
print(df.describe())


#checking unique values in categorical columns

print("\n---Unique values in categorical columns---")

print("\nproduct Categories:")
print(df['product_category'].unique())

print("\nregion:")
print(df['region'].unique())

print("\npayment Method:")
print(df['payment_method'].unique())


# Category-wise sales analysis

print("\n--- Product Category Analysis ---")
print(df.groupby("product_category")["revenue"].agg(["count", "sum", "mean"]))

print("\n--- Region-wise Sales Analysis ---")
print(df.groupby("region")["revenue"].agg(["count", "sum", "mean"]))

print("\n--- Payment Method Analysis ---")
print(df.groupby("payment_method")["revenue"].agg(["count", "sum", "mean"]))


# Revenue by Product Category

category_revenue = df.groupby("product_category")["revenue"].sum()

plt.figure(figsize=(8, 5))
category_revenue.plot(kind="bar")

plt.title("Revenue by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Revenue")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Revenue by Region

region_revenue = df.groupby("region")["revenue"].sum()

plt.figure(figsize=(8, 5))
region_revenue.plot(kind="bar")

plt.title("Revenue by Region")
plt.xlabel("Region")
plt.ylabel("Total Revenue")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



# --- Customer Analysis ---

print("\n--- Customer Analysis ---")

# Customer-wise revenue
customer_revenue = df.groupby("customer_id")["revenue"].sum().sort_values(ascending=False)

print("\nTop 10 Customers by Revenue:")
print(customer_revenue.head(10))

# Customer-wise order count
customer_orders = df.groupby("customer_id")["order_id"].count().sort_values(ascending=False)

print("\nTop 10 Customers by Number of Orders:")
print(customer_orders.head(10))

# Average revenue per customer
avg_customer_revenue = customer_revenue.mean()

print("\nAverage Revenue per Customer:")
print(round(avg_customer_revenue, 2))

# Top 10 customers chart
plt.figure(figsize=(10, 5))
customer_revenue.head(10).plot(kind="bar")
plt.title("Top 10 Customers by Revenue")
plt.xlabel("Customer ID")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Customer-wise order count
customer_orders = df.groupby("customer_id")["order_id"].count().sort_values(ascending=False)

print("\nTop 10 Customers by Number of Orders:")
print(customer_orders.head(10))

# Customer spending distribution
plt.figure(figsize=(8, 5))
plt.hist(customer_revenue, bins=20)
plt.title("Customer Revenue Distribution")
plt.xlabel("Total Revenue per Customer")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


# --- Customer Segmentation ---

print("\n--- Customer Segmentation ---")

# Create customer summary
customer_summary = df.groupby("customer_id").agg(
    total_revenue=("revenue", "sum"),
    total_orders=("order_id", "count"),
    avg_order_value=("revenue", "mean")
)

# Segment customers based on revenue
customer_summary["segment"] = pd.cut(
    customer_summary["total_revenue"],
    bins=[0, 3000, 7000, float("inf")],
    labels=["Low Value", "Medium Value", "High Value"]
)

print("\nCustomer Segmentation:")
print(customer_summary["segment"].value_counts())

# Segment revenue
segment_revenue = customer_summary.groupby(
    "segment", observed=True
)["total_revenue"].sum()

print("\nRevenue by Customer Segment:")
print(segment_revenue)

# Segment chart
plt.figure(figsize=(8, 5))
segment_revenue.plot(kind="bar")
plt.title("Revenue by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Revenue")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Customer Category Analysis ---

print("\n--- Customer Category Analysis ---")

customer_category = df.groupby(
    "product_category"
)["revenue"].sum().sort_values(ascending=False)

print("\nRevenue by Product Category:")
print(customer_category)

# Customer category chart
plt.figure(figsize=(8, 5))
customer_category.plot(kind="bar")
plt.title("Customer Revenue by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Revenue")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ---------------- Correlation Analysis ----------------

print("\n--- Correlation Analysis ---")

numeric_cols = [
    'quantity',
    'unit_price',
    'discount',
    'delivery_days',
    'customer_rating',
    'revenue'
]

correlation = df[numeric_cols].corr()

print(correlation)

# Correlation heatmap
plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ---------------- Time-Based Analysis ----------------

print("\n--- Time-Based Analysis ---")

# Convert order_date to datetime
df['order_date'] = pd.to_datetime(df['order_date'],
    dayfirst=True)

# Monthly revenue
monthly_revenue = df.groupby(
    df['order_date'].dt.to_period('M')
)['revenue'].sum()

print("\nMonthly Revenue:")
print(monthly_revenue)

# Monthly order count
monthly_orders = df.groupby(
    df['order_date'].dt.to_period('M')
)['order_id'].count()

print("\nMonthly Orders:")
print(monthly_orders)

# Revenue trend chart
plt.figure(figsize=(10, 5))

plt.plot(
    monthly_revenue.index.astype(str),
    monthly_revenue.values,
    marker='o'
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Monthly orders chart
plt.figure(figsize=(10, 5))

plt.plot(
    monthly_orders.index.astype(str),
    monthly_orders.values,
    marker='o'
)

plt.title("Monthly Order Trend")
plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()