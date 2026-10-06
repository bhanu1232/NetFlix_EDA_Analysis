import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set aesthetic style for seaborn
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

# Create output directory for plots
os.makedirs("plots", exist_ok=True)

print("Starting Netflix Dataset EDA...")

# 1. Load Data
df = pd.read_csv("netflix_titles.csv")
print(f"Dataset loaded successfully. Shape: {df.shape}")

# 2. Data Cleaning & Preparation
print("Cleaning data...")
# Handle missing values
df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')
df['date_added'] = df['date_added'].fillna(df['date_added'].mode()[0])
df['rating'] = df['rating'].fillna(df['rating'].mode()[0])

# Standardize Dates
df['date_added'] = pd.to_datetime(df['date_added'].str.strip())
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month_name()

# Standardize Durations
# Extract duration for movies
movies_df = df[df['type'] == 'Movie'].copy()
movies_df['duration_minutes'] = movies_df['duration'].str.replace(' min', '').astype(float)

# Extract seasons for TV shows
shows_df = df[df['type'] == 'TV Show'].copy()
shows_df['seasons'] = shows_df['duration'].str.extract('(\d+)').astype(float)

print("Data cleaning completed.")

# 3. Exploratory Data Analysis (EDA) & Visualizations

# Viz 1: Content Type Distribution (Movies vs TV Shows)
plt.figure(figsize=(8, 6))
type_counts = df['type'].value_counts()
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=140, colors=['#e50914', '#564d4d'], explode=(0.1, 0))
plt.title("Distribution of Content Type (Movies vs TV Shows)")
plt.savefig("plots/1_content_type_distribution.png", bbox_inches='tight')
plt.close()
print(f"Key Trend: Movies form about {type_counts['Movie'] / len(df) * 100:.1f}% of the catalog.")

# Viz 2: Top 10 Countries
plt.figure(figsize=(12, 6))
# Filter out Unknown and split multiple countries
countries = df[df['country'] != 'Unknown'].copy()
countries_list = countries['country'].str.split(',').explode().str.strip()
top_countries = countries_list.value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index, palette="Reds_r")
plt.title("Top 10 Content Producing Countries")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.savefig("plots/2_top_10_countries.png", bbox_inches='tight')
plt.close()
print(f"Key Trend: The United States and India are the top producing countries.")

# Viz 3: Content Addition Over the Years
plt.figure(figsize=(12, 6))
content_by_year = df.groupby(['year_added', 'type']).size().unstack().fillna(0)
content_by_year.plot(kind='line', marker='o', color=['#e50914', '#564d4d'], figsize=(12, 6))
plt.title("Content Added Over the Years")
plt.xlabel("Year Added")
plt.ylabel("Number of Titles")
plt.grid(True)
plt.savefig("plots/3_content_added_over_years.png", bbox_inches='tight')
plt.close()

# Viz 4: Content Addition by Month
plt.figure(figsize=(12, 6))
months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
df['month_added'] = pd.Categorical(df['month_added'], categories=months_order, ordered=True)
sns.countplot(x='month_added', hue='type', data=df, palette=['#e50914', '#564d4d'])
plt.title("Content Added by Month")
plt.xlabel("Month")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig("plots/4_content_added_by_month.png", bbox_inches='tight')
plt.close()

# Viz 5: Rating Distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='rating', data=df, order=df['rating'].value_counts().index, palette="rocket")
plt.title("Overall Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig("plots/5_rating_distribution.png", bbox_inches='tight')
plt.close()

# Viz 6: Rating Distribution by Content Type
plt.figure(figsize=(14, 6))
sns.countplot(x='rating', hue='type', data=df, order=df['rating'].value_counts().index, palette=['#e50914', '#564d4d'])
plt.title("Rating Distribution by Content Type")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig("plots/6_rating_by_type.png", bbox_inches='tight')
plt.close()

# Viz 7: Top 10 Genres
plt.figure(figsize=(12, 6))
genres = df['listed_in'].str.split(',').explode().str.strip()
top_genres = genres.value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index, palette="mako")
plt.title("Top 10 Genres on Netflix")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
plt.savefig("plots/7_top_10_genres.png", bbox_inches='tight')
plt.close()

# Viz 8: Movie Duration Distribution
plt.figure(figsize=(10, 6))
sns.histplot(movies_df['duration_minutes'], bins=50, kde=True, color='#e50914')
plt.title("Distribution of Movie Durations")
plt.xlabel("Duration (minutes)")
plt.ylabel("Frequency")
plt.savefig("plots/8_movie_duration_distribution.png", bbox_inches='tight')
plt.close()

# Viz 9: TV Show Seasons Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='seasons', data=shows_df, palette="viridis")
plt.title("Distribution of TV Show Seasons")
plt.xlabel("Number of Seasons")
plt.ylabel("Count")
plt.xlim(-0.5, 9.5) # Focusing on up to 10 seasons for better visibility
plt.savefig("plots/9_tv_show_seasons_distribution.png", bbox_inches='tight')
plt.close()

# Viz 10: Top 10 Directors
plt.figure(figsize=(12, 6))
directors = df[df['director'] != 'Unknown']['director'].str.split(',').explode().str.strip()
top_directors = directors.value_counts().head(10)
sns.barplot(x=top_directors.values, y=top_directors.index, palette="magma")
plt.title("Top 10 Directors on Netflix")
plt.xlabel("Number of Titles")
plt.ylabel("Director")
plt.savefig("plots/10_top_10_directors.png", bbox_inches='tight')
plt.close()

# Viz 11: Release Year Distribution (Last 20 Years)
plt.figure(figsize=(12, 6))
recent_years = df[df['release_year'] >= 2000]
sns.countplot(x='release_year', hue='type', data=recent_years, palette=['#e50914', '#564d4d'])
plt.title("Content Releases Over the Last 20+ Years")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig("plots/11_release_year_distribution.png", bbox_inches='tight')
plt.close()

print("EDA successfully completed! All 11 visualizations have been saved in the 'plots' directory.")
