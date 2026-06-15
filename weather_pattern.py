import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv(
    r"C:\Users\JC\OneDrive\Desktop\Summer internship\project-4\indian_weather.csv"
)

# --------------------
# DATA CLEANING
# --------------------

df = df.dropna()
df = df.drop_duplicates()

print("Dataset Preview:")
print(df.head())

print("\nColumns:")
print(df.columns)

# Convert datetime column
df['forecast_datetime'] = pd.to_datetime(df['forecast_datetime'])

# --------------------
# ANALYSIS
# --------------------

print("\n----- ANALYSIS -----")

print("Total Records:", len(df))

print("Average Temperature:", round(df['temperature_c'].mean(), 2))

print("Maximum Temperature:", df['temperature_c'].max())

print("Minimum Temperature:", df['temperature_c'].min())

print("Most Common Weather:")
print(df['weather'].mode()[0])

# --------------------
# VISUALIZATION
# --------------------

# 1 Temperature Trend
plt.figure(figsize=(8, 5))
plt.plot(df['forecast_datetime'], df['temperature_c'])
plt.title("Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.show()

# 2 Weather Distribution
weather_count = df['weather'].value_counts()

weather_count.plot(kind='bar')
plt.title("Weather Distribution")
plt.xlabel("Weather")
plt.ylabel("Count")
plt.show()

# 3 Humidity Distribution
df['relative_humidity'].plot(kind='hist', bins=10)
plt.title("Humidity Distribution")
plt.xlabel("Relative Humidity")
plt.show()

# 4 Wind Speed Trend
plt.figure(figsize=(8, 5))
plt.plot(df['forecast_datetime'], df['wind_speed'])
plt.title("Wind Speed Trend")
plt.xlabel("Date")
plt.ylabel("Wind Speed")
plt.xticks(rotation=45)
plt.show()

# 5 Cloud Cover Distribution
cloud_cover = df['cloud_cover'].value_counts()

cloud_cover.plot(kind='pie', autopct='%1.1f%%')
plt.title("Cloud Cover Distribution")
plt.ylabel("")
plt.show()

# 6 Temperature Distribution
df['temperature_c'].plot(kind='hist', bins=15)
plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.show()

# 7 Top 10 Cities by Average Temperature
city_temp = df.groupby('city')['temperature_c'].mean().sort_values(ascending=False).head(10)

city_temp.plot(kind='bar')
plt.title("Top 10 Cities by Average Temperature")
plt.xlabel("City")
plt.ylabel("Average Temperature")
plt.show()

# --------------------
# SIMPLE PREDICTION
# --------------------

prediction = df['temperature_c'].mean()

print("\n----- PREDICTION -----")
print("Predicted Future Temperature:", round(prediction, 2), "°C")

# --------------------
# DASHBOARD
# --------------------

print("\n----- DASHBOARD -----")

print("Average Temperature:", round(df['temperature_c'].mean(), 2))

print("Maximum Temperature:", df['temperature_c'].max())

print("Minimum Temperature:", df['temperature_c'].min())

print("Average Humidity:", round(df['relative_humidity'].mean(), 2))

print("Average Wind Speed:", round(df['wind_speed'].mean(), 2))

print("Most Common Weather:", df['weather'].mode()[0])

print("Predicted Temperature:", round(prediction, 2))