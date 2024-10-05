#Mental Fitness Tracker

Mental Fitness Tracker
This project is a mental fitness tracker that incorporates data analysis, visualization, and machine learning techniques to analyze mental health-related data and track users' mood and stress levels. The project demonstrates merging datasets, performing data cleaning, visualizing data trends, and applying machine learning models (Linear Regression, Random Forest) for mental fitness prediction. Additionally, AI sentiment analysis using a BERT model is integrated to track mood and stress levels over time.

Features
Data Analysis and Visualization: This project analyzes two mental health-related datasets, visualizes trends, and examines correlations.

Machine Learning Models: Linear Regression and Random Forest are applied for predicting mental fitness using the dataset.

AI Sentiment Analysis: Sentiment prediction for mood tracking using BERT-based AI model.

Interactive Visualizations: Line plots, heatmaps, and pairwise plots created using Seaborn, Matplotlib, and Plotly libraries.

User Interaction: Mood and stress tracking feature with AI-based sentiment prediction, and visualization of the user's mood and stress trends over time.


Libraries Used

NumPy: For linear algebra operations.

Pandas: For data processing and CSV file I/O.

Matplotlib & Seaborn: For visualizing trends and correlations in data.

Plotly: For interactive plots.

Sklearn: For data splitting and building machine learning models.

Transformers (HuggingFace): For AI-based sentiment prediction using a pre-trained BERT model.

Torch: For working with the BERT model in PyTorch.


Dataset Details
prevalence-by-mental-and-substance-use-disorder.csv: Contains data on the prevalence of mental and substance use disorders across different countries and years.
mental-and-substance-use-as-share-of-disease.csv: Contains data on mental and substance use disorders as a share of overall disease burden.


Key Steps in the Project

Importing Libraries and Mounting Google Drive:
Libraries for data analysis, visualization, and machine learning are imported.
Google Drive is mounted in Colab to access datasets stored in the drive.


Data Loading and Merging:
Two datasets are loaded and merged for further analysis.


Data Cleaning:
Dropping unnecessary columns and handling missing values.

Data Visualization:
Heatmaps: For analyzing correlations between various mental health conditions.
Pairwise Relationships: For visualizing relationships between different variables.
Yearwise Variations: Using a line plot to display the variations in mental fitness across different countries.

Machine Learning Models:
Linear Regression and Random Forest Regressor: Applied to predict mental fitness. Models are evaluated based on MSE, RMSE, and R2 score for both training and testing sets.
AI-based Sentiment Analysis for Mood Tracking:

Utilizes a BERT-based sentiment analysis model to predict the sentiment of a user’s mood.
Prompts users to enter their daily mood and stress level, and stores the data for visualization.
Data Visualization for User's Mood and Stress:

Visualizes the user’s mood and stress level data with date annotations using Matplotlib.
Instructions for Use
Running the Program:

Mount Google Drive to access the datasets.
Run the Python script to merge, clean, and analyze the datasets.
Evaluate the machine learning models for mental fitness prediction.
Use the track_mood_with_ai() function to input your mood and stress levels, and track the sentiment using the AI-based model.
Visualize the trends using visualize_mood_data() to observe how your mood and stress levels change over time.
Sentiment Analysis:

The AI-based sentiment prediction will classify the mood as Positive, Neutral, or Negative.


Model Training:
The Linear Regression and Random Forest models are trained and evaluated based on mental fitness data, allowing the prediction of mental fitness trends.


Visualization:
After tracking mood and stress levels, visualize them to gain insights into the patterns of mental health.
Sample Code Execution

# Run the mood tracking function
track_mood_with_ai()

# Visualize the tracked data
visualize_mood_data()


Requirements
To run this project, ensure you have the following libraries installed:
pip install numpy pandas matplotlib seaborn plotly scikit-learn transformers torch

Acknowledgements
Datasets: The datasets used in this project are sourced from publicly available mental health statistics.
AI Model: The sentiment analysis model is based on the pre-trained BERT-base-uncased model from HuggingFace's Transformers library.
This project demonstrates the integration of data analysis, machine learning, and AI-based sentiment analysis to track and improve mental fitness
