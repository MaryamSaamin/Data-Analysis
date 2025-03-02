#%%
print("hello world")

# %%
# Import necessary libraries
from pyspark import SparkContext
from pyspark.sql import SQLContext

# Create a SparkContext
sc = SparkContext(appName="Titanic Dataset")

# Create an SQLContext
sqlContext = SQLContext(sc)

# Load the Titanic dataset
df = sqlContext.read.csv("data/titanic.csv", header=True, inferSchema=True)

# Count the number of lines
print("Number of lines:", df.count())

# Print a few lines of data
df.show(5)

# Don't forget to stop the SparkContext when done
sc.stop()



# %%

from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Shopping Trends Analysis") \
    .getOrCreate()

# Load the dataset into a DataFrame
df = spark.read.csv("data/shopping_trends/shopping_trends.csv", header=True, inferSchema=True)


# %%

from pyspark.sql import functions as F

# Assuming df is your DataFrame containing shopping data

# Convert 'Purchase Amount (USD)' to numeric type
df = df.withColumn("Purchase Amount (USD)", df["Purchase Amount (USD)"].cast("decimal(10, 2)"))


# %%

frequency_total_purchases = df.groupBy("Age", "Customer ID") \
    .agg(F.sum("Purchase Amount (USD)").alias("Total Purchases"))

# Show the resulting DataFrame
frequency_total_purchases.show()

# %%
# Compute statistics for total purchase amounts across customers
purchase_statistics = df.groupBy().agg(
    F.mean("Purchase Amount (USD)").alias("Mean"),
    F.expr("percentile_approx(`Purchase Amount (USD)`, 0.5)").alias("Median"),
    F.stddev("Purchase Amount (USD)").alias("Standard Deviation"),
    F.expr("percentile_approx(`Purchase Amount (USD)`, 0.25)").alias("Q1"),
    F.expr("percentile_approx(`Purchase Amount (USD)`, 0.75)").alias("Q3")
)

# Show the resulting DataFrame
purchase_statistics.show()

# %%
# Registering the DataFrame as a temporary view
frequency_total_purchases.createOrReplaceTempView("customer_total_purchases")

# Now you can use SQL queries on this view
# Example SQL query to select data from the view
result = spark.sql("SELECT * FROM customer_total_purchases")

# Display the result
result.show()


# %%
# SQL query to select the top 10 customers with the highest total purchases
top_10_customers_query = """
    SELECT 
        `Age`, 
        `Customer ID`, 
        `Total Purchases`
    FROM 
        customer_total_purchases
    ORDER BY 
        `Total Purchases` DESC
    LIMIT 10
"""

# Execute the SQL query
top_10_customers = spark.sql(top_10_customers_query)

# Show the result
top_10_customers.show()

# %%
# Registering the DataFrame as a temporary view
df.createOrReplaceTempView("shopping_data")

# SQL query to calculate the average purchase amount per age for all customers, grouped by gender
average_purchase_per_age_query = """
    SELECT 
        `Age`,
        `Gender`,
        AVG(`Purchase Amount (USD)`) AS Average_Purchase_Amount
    FROM 
        shopping_data
    GROUP BY 
        `Age`, 
        `Gender`
    ORDER BY
        `Age`, 
        `Gender`
"""

# Execute the SQL query
average_purchase_per_age = spark.sql(average_purchase_per_age_query)

# Show the result
average_purchase_per_age.show()

# %%
from pyspark.sql import functions as F

# Grouping the DataFrame by Age and counting the number of purchases
total_purchases_per_age = df.groupBy("Age").count().orderBy("Age")

# Renaming the count column for clarity
total_purchases_per_age = total_purchases_per_age.withColumnRenamed("count", "Total_Purchases")

# Show the result
total_purchases_per_age.show()

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Convert PySpark DataFrame to Pandas DataFrame
total_purchases_pd = total_purchases_per_age.toPandas()

# Plotting the results as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(total_purchases_pd['Age'], total_purchases_pd['Total_Purchases'], color='skyblue')
plt.xlabel('Age')
plt.ylabel('Total Purchases')
plt.title('Total Number of Purchases Made in Each Age')
plt.xticks(total_purchases_pd['Age'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
# Plotting the results as a line chart
plt.figure(figsize=(10, 6))
plt.plot(total_purchases_pd['Age'], total_purchases_pd['Total_Purchases'], marker='o', color='orange', linestyle='-')
plt.xlabel('Age')
plt.ylabel('Total Purchases')
plt.title('Total Number of Purchases Made in Each Age (Trend)')
plt.xticks(total_purchases_pd['Age'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
