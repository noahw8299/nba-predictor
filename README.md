# NBA Game Predictor

Welcome to the NBA Game Predictor! This project aims to leverage machine learning to predict the outcomes of NBA games based on historical data and various statistical features. The project showcases my skills in data science, machine learning, web development, and software engineering.

## Table of Contents
1. [Project Overview](#nba-game-predictor)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Data Collection](#data-collection)
8. [Machine Learning Model](#machine-learning-model)
9. [Streamlit Web App](#streamlit-web-app)
10. [Deployment](#deployment)
11. [Future Improvements](#future-improvements)
12. [License](#license)

## Features

- **Data Scraping:** Collects data from various NBA-related websites.
- **Data Preprocessing:** Cleans and preprocesses the scraped data for analysis.
- **Model Training:** Trains machine learning models to predict game outcomes.
- **Interactive Web Interface:** Provides an interactive web interface using Streamlit for users to input game details and get predictions.
- **Deployment:** Deployable on Heroku for easy access.

## Technologies Used

- **Python:** The main programming language used.
- **Pandas:** For data manipulation and analysis.
- **Scikit-learn:** For machine learning model development.
- **Jupyter Notebooks:** For data exploration and prototyping.
- **Streamlit:** For building the web application.
- **BeautifulSoup:** For web scraping.
- **Git:** For version control.
- **Heroku:** For deployment.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/nba-predictor.git
    cd nba-predictor
    ```

2. **Set up a virtual environment and activate it:**

    ```bash
    python3 -m venv nba_env
    source nba_env/bin/activate  # On Windows use `nba_env\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Streamlit App

1. **Start the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

2. **Open your browser and go to `http://localhost:8501` to interact with the app.**

## Project Structure

nba-predictor/
├── app/
│ ├── app.py # Main application file for Streamlit
│ ├── predict_model.pkl # Trained model for predictions
│ ├── predict_page.py # Prediction page script
│ ├── images/ # Directory for storing images
├── model/
│ ├── predict/ # Directory for prediction related scripts and notebooks
│ ├── scrape/ # Directory for scraping related scripts and notebooks
├── nba_env/ # Virtual environment directory (not tracked by git)
├── requirements.txt # List of dependencies
├── Procfile # Heroku deployment file
├── README.md # Project README file


## Data Collection

The data is collected from various NBA-related websites using web scraping techniques. The scripts in the `scrape` directory are responsible for fetching and storing this data.

### Scraping Scripts

- `scrape/get_data.ipynb`: Notebook for scraping game data.
- `scrape/parse_data.ipynb`: Notebook for parsing and cleaning scraped data.

## Machine Learning Model

The machine learning models are developed using scikit-learn. The models are trained on historical NBA game data to predict future game outcomes.

### Notebooks and Scripts

- `model/predict/nba-predict-spread.ipynb`: Notebook for model development and training.
- `model/predict/nba-predict-v2.ipynb`: Updated model training notebook.

## Streamlit Web App

The web app is built using Streamlit, providing an interactive interface for users to input game details and view predictions.

### Application Files

- `app/app.py`: Main application file.
- `app/predict_page.py`: Script for the prediction page.

## Deployment

The project can be deployed on Heroku for easy access. Follow these steps to deploy:

1. **Login to Heroku and create a new app:**

    ```bash
    heroku login
    heroku create your-app-name
    ```

2. **Push the code to Heroku:**

    ```bash
    git push heroku main
    ```

3. **Open your app:**

    ```bash
    heroku open
    ```

## Future Improvements

- **Model Improvement:** Enhance the model accuracy by incorporating advanced machine learning techniques.
- **Data Visualization:** Add more visualizations to the web app to better represent the data and predictions.
- **User Authentication:** Implement user authentication for personalized predictions and history tracking.