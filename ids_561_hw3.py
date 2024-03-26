# -*- coding: utf-8 -*-

from google.colab import drive

drive.mount('/content/drive')

!pip install pyspark

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, countDistinct, when
from pyspark.sql import functions as F

# Create a SparkSession object
spark = SparkSession.builder \
    .appName("Amazon Tweets Analysis") \
    .master("local[2]") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

"""**Step 1:** Find out the users who are active in at least five listed days (i.e., created posts in at least 5 days) in Amazon_Responded_Oct05.csv and save their “user_screen_name” and “user_id_str” in the dataframe “daily_active_users” (see below). Report how many active users you find."""

# Load the data
df = spark.read.csv('/content/drive/My Drive/IDS561 Analytics for Big Data/HW3/Amazon_Responded_Oct05.csv', header=True, multiLine=True, escape="\"")
df.show()

#Check the original date form in the column "tweet_created_at"
df.select("tweet_created_at").show(5, truncate=False)

#Set the legacy time parser policy
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

#Transform the date form to YYYY/MM/DD and put them into a newly created column "date"
from pyspark.sql import functions as F

df = df.withColumn(
    "date",
    F.from_unixtime(
        F.unix_timestamp("tweet_created_at", "EEE MMM dd HH:mm:ss ZZZZZ yyyy"),
        "yyyy/MM/dd"
    )
)

#Show the newly created column "date"
df.select("date").show(5, truncate=False)

# Group by 'user_screen_name' and 'user_id_str', and count distinct 'date'
active_users = df.groupBy('user_screen_name', 'user_id_str').agg(countDistinct('date').alias('active_days'))

# Filter users who are active in at least five listed days
daily_active_users = active_users.filter(col('active_days') >= 5)

# Show the dataframe
daily_active_users.show(50)

# Print the number of active users
print("Number of active users: ", daily_active_users.count())

"""**Step 2:** A company would like to conduct an A/B test on Twitter. The experiment.txt file includes the user_id_str they selected as potential experiment targets. Please create a dataframe “experiment_user” to document the selected user id and whether they are active users (join the dataframe from step 1). Then calculate the percentage of active user and print out the result."""

# Load the experiment data
experiment_df = spark.read.text('/content/drive/My Drive/IDS561 Analytics for Big Data/HW3/experiment.txt').toDF("user_id_str")

# Create a dataframe to document the selected user id and whether they are active users
experiment_user = experiment_df.join(daily_active_users, on="user_id_str", how="left")

# Create a new column 'whether_active' to indicate whether the user is active
experiment_user = experiment_user.withColumn("whether_active", when(col("active_days").isNull(), "no").otherwise("yes"))

# Shows only desired columns ("user_id_str" and "whether_active")
experiment_user = experiment_user.drop("user_screen_name", "active_days")

experiment_user.show(50)

# Calculate the percentage of active users
active_users_count = experiment_user.filter(experiment_user.whether_active == "yes").count()
total_users_count = experiment_user.count()
percentage_active = (active_users_count / total_users_count) * 100

# Print out the result
print("Percentage of active users: ", percentage_active)

"""**Step 3:** The company provided their revised experiment target list in final_experiment.csv file. Compared with the former experiment.txt file, they removed several users and added a new column “info” to indicate whether the user is female (F) or male (M). Fill in the remaining columns by joining the dataframes you got from step 1&2 together and save the result in a dataframe “final_experiment”, and describe your join steps briefly."""

# Load the final_experiment.csv file
final_experiment_df = spark.read.csv('/content/drive/My Drive/IDS561 Analytics for Big Data/HW3/final_experiment_data.csv', header=True)

final_experiment_df.show()
final_experiment_df.count()

#Drop the columns "whether_active" and "user_screen_name" which will be added back later after joining other tables
final_experiment_df = final_experiment_df.drop(final_experiment_df.whether_active, final_experiment_df.user_screen_name)

final_experiment_df.show()
final_experiment_df.count()

# Create a new table by joining target users in "final_experiment_df" and "whether_active" from "experiment_user"
df_Temp = final_experiment_df.join(experiment_user, on="user_id_str", how="left")

df_Temp.show()
df_Temp.count()

# Use outer join to make sure all of the new targets are included
# We'll filter out the removed targets from experiment_users (whose "info" is NULL)
df_Temp = df_Temp.join(daily_active_users.select("user_id_str", "user_screen_name"), on="user_id_str", how="outer")

df_Temp.show()
df_Temp.count()

# For inactive users that cannot be found in “daily_active_users”, “user_screen_name” will be filled with “Not found”
df_Temp = df_Temp.withColumn("user_screen_name", F.when(F.col("whether_active") == "no", "Not found").otherwise(F.col("user_screen_name")))

# Drop rows where 'info' column is NULL
df_Temp = df_Temp.filter(df_Temp.info.isNotNull())

df_Temp.show()
df_Temp.count()

# Save the three dataframe as CSV files
daily_active_users.write.csv('daily_active_users.csv', header=True)

experiment_user.write.csv('experiment_user.csv', header=True)

final_experiment_df.write.csv('Final_experiment.csv', header=True)