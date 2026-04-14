# Restaurant-Recommendation-System

DineAdvisor is a Flask-based restaurant recommendation web app that suggests similar restaurants using a content-based filtering approach.

The app recommends restaurants by comparing cuisine similarity (TF-IDF + cosine similarity), then applies optional filters for cuisine, budget, and minimum rating.

Features
Content-based restaurant recommendation engine
Case-insensitive restaurant search with partial-name fallback
Filtering by:
preferred cuisine
maximum budget per person
minimum rating
Top results sorted by rating
User-friendly error handling for no-match scenarios
Modern responsive dark glassmorphism UI
Tech Stack
Python
Flask
Pandas
Scikit-learn
HTML + CSS (Jinja templates)
