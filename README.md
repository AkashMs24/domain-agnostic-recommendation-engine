📌 Domain-Agnostic Recommendation Engine

A production-stable, domain-agnostic recommendation system that compares classical TF-IDF similarity with semantic-style recommendations, supports cold-start scenarios, and allows users to upload their own datasets — all wrapped in a polished Streamlit UI.

🚀 Project Overview

Recommendation systems are commonly demonstrated using fixed product or movie datasets.
This project goes beyond that by building a reusable recommendation engine that works for any text-based dataset, such as:

E-commerce products
Research papers
Career paths
Courses or skills
Academic resources

The system focuses on stability, explainability, and usability, making it suitable for real-world deployment scenarios.

✨ Key Features

🔁 Dual Recommendation Engines
TF-IDF (classical, keyword-based)
Semantic-style similarity (stable, production-safe)

❄️ Cold-Start Friendly
No user history required
Works purely from text input

📤 Upload Your Own Dataset
Any CSV with title and description
Instantly generates recommendations

📊 Confidence Visualization

Similarity scores shown with progress bars
⭐ Save / Pin Recommendations
Session-based memory for shortlisted items

🧠 Explainable Results

Clear reasoning behind recommendations

🎨 Polished UI

Dark theme
Clean layout
Product-style presentation

🏗️ System Architecture
Offline (Model Preparation)
TF-IDF model trained on product descriptions
Similarity matrix and embeddings saved as .pkl files

Online (Streamlit App)
Loads precomputed models
Accepts user queries and uploaded datasets
Computes similarity scores
Displays ranked recommendations
https://domain-agnostic-recommendation-engine-5zsyl9hfwhj5myenx45k5k.streamlit.app/

⚠️ Design Decision:
To ensure runtime stability on Windows and Streamlit, heavy transformer models are not loaded during inference.
This mirrors real-world production systems where embeddings are precomputed offline.

🛠️ Tech Stack

Python
Pandas & NumPy
Scikit-learn
Streamlit
TF-IDF (NLP)
Cosine Similarity
