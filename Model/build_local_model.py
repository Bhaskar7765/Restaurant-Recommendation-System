import pandas as pd
import numpy as np
import pickle
import re
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading local dataset...")
df = pd.read_csv('../Flask/restaurant1.csv', encoding='utf-8')
print(f"Dataset Shape: {df.shape}")

# Pre-clean restaurant names 
def clean_string(text):
    text = str(text)
    text = text.replace('Ã©', 'e').replace('Ã¨', 'e').replace('Ã\xad', 'i')
    text = text.replace('Ã¢', 'a').replace('Ã´', 'o').replace('Ã»', 'u').replace('Ã±', 'n')
    text = text.replace('Â', '').replace('©', '')
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

df['name'] = df['name'].apply(clean_string)

print("Applying data preprocessing...")
df.drop(['online_order', 'book_table', 'listed_in(type)', 'listed_in(city)'], axis=1, inplace=True, errors='ignore')

# Clean rate
df['rate'] = df['rate'].astype(str).apply(lambda x: x.replace('/5', '').strip())
df['rate'] = df['rate'].replace(['NEW', '-'], np.nan)
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
df['rate'].fillna(df['rate'].mean(), inplace=True)


df.drop_duplicates(subset='name', keep='first', inplace=True)

# Sample if too large
if len(df) > 4000:
    df = df.sample(n=4000, random_state=42).reset_index(drop=True)
print(f"Preprocessed Shape: {df.shape}")

print("Feature engineering...")
def create_soup(x):
    cuisines = str(x['cuisines']).replace(', ', ' ')
    rest_type = str(x.get('rest_type', '')).replace(', ', ' ')
    city = str(x.get('listed_in(city)', '')).replace(' ', '')
    return f"{cuisines} {rest_type} {city}"

df['soup'] = df.apply(create_soup, axis=1)
df['soup'] = df['soup'].apply(lambda x: re.sub('[^a-zA-Z ]', ' ', x).lower())

print("Cosine similarity...")
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(df.index, index=df['name']).drop_duplicates()

print("Saving model...")
model_data = {
    'similarity_matrix': cosine_sim,
    'indices': indices,
    'restaurant_data': df[['name', 'cuisines', 'location', 'rate', 'approx_cost(for two people)']]
}

with open('../Flask/restaurant.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Success! restaurant.pkl saved to Flask/. Restart app1.py to load.")
