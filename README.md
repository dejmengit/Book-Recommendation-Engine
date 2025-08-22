# 📚 **Book Recommendation Engine**

This project is a hands-on implementation of a Book Recommender System built with Python and data science techniques.
It guides you from data collection and cleaning through machine learning methods to building a working prototype recommender system.


##  **Project Overview**

This project includes:

 Data Collection – Using OpenLibrary API to gather book metadata.  
 Data Cleaning & Preparation – Creating a structured dataset.  
 Machine Learning Models  
 Content-Based Filtering TF-IDF
 Collaborative Filtering (user–item interactions)  
 Hybrid System (weighted mix of both approaches)  
 Evaluation & Visualizations  
 Prototype UI – Built with Streamlit for easy demo.  

##  **Tech Stack**

Python
Pandas / NumPy  
Scikit-learn – TF-IDF
SciPy  
Matplotlib
Streamlit  

## 📊 **Dataset**

Collected via OpenLibrary API  

~1600 books with fields:

- Title
- Main Author
- Genres
- First Publish Year
- Cover Image URL
- Work Key

## 🤖 **Recommender Approaches**
1️⃣ Content-Based Filtering

Uses TF-IDF on book metadata (title + author + subjects).
Computes cosine similarity between books.

2️⃣ Collaborative Filtering

Builds a user–item interaction matrix from likes/clicks.
Learns item–item similarity from co-liked books.

3️⃣ Hybrid Model


## 🖥️ **How to Run the Prototype**

Clone this repo or download the files.

Install requirements:
pip install -r requirements_streamlit.txt


Launch the app:
streamlit run streamlit_app.py

In the app sidebar:
- Enter your User Name
- Choose: Cold-start (pick favorites) OR Use my past likes
- Browse recommendations & click 👍 to log likes

## 📈 **Evaluation & Visualizations**

📊 Visual insights include:
- Top 15 Authors
- Top 15 Subjects
- Books per Decade
- Precision Comparison (CF vs Content)

## ✨ **Future Improvements**

🔗 Integrate Google Books  
⭐ Include user ratings & reviews.  
☁️ Deploy app on Streamlit Cloud  
🔍 Add advanced filters (genre, author, publication year).  

##  **Credits**

Developed as part of a Book Recommendation System assignment, practicing:  
Web scraping & API usage  
Data preprocessing  
Unsupervised learning  
Recommender system design  
