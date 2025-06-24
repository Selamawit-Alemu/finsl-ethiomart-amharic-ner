import pandas as pd

# Assuming df with columns: vendor_name, username, timestamp, text, views, price_extracted

# 1. Calculate posting frequency per vendor (posts per week)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['week'] = df['timestamp'].dt.to_period('W')
posts_per_week = df.groupby(['vendor_name', 'week']).size().groupby('vendor_name').mean()

# 2. Average views per post per vendor
avg_views = df.groupby('vendor_name')['views'].mean()

# 3. Average price point (assuming price_extracted is numeric)
avg_price = df.groupby('vendor_name')['price_extracted'].mean()

# 4. Top performing post (highest views) per vendor
top_posts = df.loc[df.groupby('vendor_name')['views'].idxmax()][['vendor_name', 'text', 'views']]

# 5. Compose final score (example weighting)
lending_score = 0.5 * avg_views + 0.5 * posts_per_week

# 6. Aggregate into scorecard
scorecard = pd.DataFrame({
    'Posts/Week': posts_per_week,
    'Avg Views/Post': avg_views,
    'Avg Price (ETB)': avg_price,
    'Lending Score': lending_score
}).reset_index()

print(scorecard.sort_values('Lending Score', ascending=False))
