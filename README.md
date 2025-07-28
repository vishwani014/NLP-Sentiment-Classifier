# NLP-Sentiment-Classifier
A Python-based web application for sentiment analysis of text data, built using NLTK's VADER model and Streamlit for a user-friendly interface. This project demonstrates proficiency in natural language processing (NLP), data preprocessing, and UI/UX design. The app analyzes text to classify sentiment as positive, negative, or neutral, with interactive visualizations for batch analysis.



# Features

- ## Single Text Analysis: 
Input a single text to receive instant sentiment classification (positive, negative, neutral) with detailed polarity scores.

- ## Batch CSV Analysis:
Upload a CSV file with a 'text' column to perform sentiment analysis on multiple texts and visualize results with a distribution plot.

- ## User-Friendly Interface: 
Features a clean, responsive UI with color-coded results (green for positive, red for negative, gray for neutral) and intuitive navigation, designed to enhance user experience.

- ## Robust Data Preprocessing: 
Implements text cleaning (e.g., removing URLs, special characters) and NLP preprocessing (tokenization, stopword removal) to ensure high-quality input for analysis.



# Installation - To run the app locally, follow these steps:

- ## Clone the Repository:
git clone https://github.com/vishwani014/NLP-Sentiment-Classifier.git
cd NLP-Sentiment-Classifier

- ## Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

- ## Install Dependencies:
pip install -r requirements.txt

- ## Run the App:
streamlit run sentiment_analysis_app.py



# Usage

- ## Single Text Analysis: 
Navigate to the "Single Text Analysis" tab, enter text (e.g., a customer review), and click "Analyze Sentiment" to view the sentiment classification and polarity scores.

- ## Batch CSV Analysis: 
In the "CSV File Analysis" tab, upload a CSV file with a 'text' column (e.g., sample_reviews.csv included in the repository) to analyze multiple texts and view a sentiment distribution plot.

- ## Sample Dataset: 
The repository includes sample_reviews.csv for testing batch analysis.



# Libraries and Tools Used - The project leverages industry-standard libraries and tools, selected for their efficiency, reliability, and relevance to NLP and UI/UX tasks:

- ## Pandas: 
Used for efficient data manipulation and CSV handling. Chosen for its robust DataFrame functionality, enabling seamless processing of batch text data, which is critical for ML workflows.

- ## NLTK (Natural Language Toolkit): 
Employed for text preprocessing (tokenization, stopword removal) and sentiment analysis via the VADER model. Selected for its beginner-friendly API and VADER’s effectiveness in analyzing short texts (e.g., social media), aligning with real-world NLP applications.

- ## Streamlit: 
Utilized to build an interactive web interface with minimal code. Chosen for its rapid prototyping capabilities and ability to create responsive, user-friendly UIs, showcasing my UI/UX design skills.

- ## Matplotlib: 
Used for creating visualizations, such as the sentiment distribution plot. Selected for its flexibility in generating static plots, a standard tool for data visualization in ML projects.

- ## Seaborn: 
Employed to enhance visualization aesthetics (e.g., count plots). Chosen for its high-level interface, which simplifies creating professional, visually appealing charts to communicate insights effectively.

- ## re (Regular Expressions): 
Used for text cleaning (e.g., removing URLs, special characters). Selected for its lightweight, built-in functionality in Python, ideal for preprocessing raw text data in NLP pipelines.



# Live Demo
View the app on Streamlit Cloud https://nlp-sentiment-classifier-cb6uzqre9qykvqmu5dna86.streamlit.app

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Contact
Vishwani Vilochana | https://www.linkedin.com/in/vishwani-vilochana-498a5b246/ | vishwani2002@gmail.com