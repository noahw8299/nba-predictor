# NBA Spread Predictor

Welcome to the NBA Spread Predictor! This project leverages machine learning to predict the point spread outcomes of NBA games based on historical data and various statistical features. The project showcases my skills in data science, machine learning, web development, and software engineering.

## Table of Contents
1. [Project Overview](#nba-spread-predictor)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Data Collection](#data-collection)
8. [Machine Learning Model](#machine-learning-model)
9. [Streamlit Web App](#streamlit-web-app)
10. [Future Improvements](#future-improvements)
11. [License](#license)

## Features

- **Data Scraping:** Collects data from various NBA-related websites.
- **Data Preprocessing:** Cleans and preprocesses the scraped data for analysis.
- **Model Training:** Trains machine learning models to predict game spreads.
- **Interactive Web Interface:** Provides an interactive web interface using Streamlit for users to input game details and get predictions.
- **Deployment:** Deployable on Streamlit Community Cloud for easy access.

## Technologies Used

- **Python:** The main programming language used.
- **Pandas:** For data manipulation and analysis.
- **Scikit-learn:** For machine learning model development.
- **Jupyter Notebooks:** For data exploration and prototyping.
- **Streamlit:** For building the web application.
- **BeautifulSoup:** For web scraping.
- **Git:** For version control.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/noahw8299/nba-spread-predictor.git
    cd nba-spread-predictor
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

### Running the Streamlit App Locally

If you wish to interact with the code on your personal device:

1. **Start the Streamlit app:**

    ```bash
    streamlit run app/app.py
    ```

2. **Open your browser and go to `http://localhost:8501` to interact with the app.**

### Interacting with the Hosted App

If you wish to see and interact with the app without setting it up locally, you can access it [here](https://nba-predictor.streamlit.app/).

## Project Structure

```plaintext
nba-spread-predictor
├── app
│   ├── app.py                # Main application file for Streamlit
│   ├── predict_page.py       # Prediction page script
│   ├── predict_model.pkl     # Trained model for predictions
│   ├── model_scaler.pkl      # Scaler used for feature scaling
│   ├── images                # Directory for storing images
│   ├── result.csv            # CSV file with historical game data
├── model
│   ├── predict               # Directory for prediction related scripts and notebooks
│   ├── scrape                # Directory for scraping related scripts and notebooks
├── nba_env                   # Virtual environment directory (not tracked by git)
├── requirements.txt          # List of dependencies
├── README.md                 # Project README file
```

## Data Collection

The data is collected from various NBA-related websites using web scraping techniques. The scripts in the `scrape` directory are responsible for fetching and storing this data.

### Scraping Scripts

- `scrape/get_data.ipynb`: Notebook for scraping game data.
- `scrape/parse_data.ipynb`: Notebook for parsing and cleaning scraped data.

## Machine Learning Model

The machine learning models are developed using scikit-learn. The models are trained on historical NBA game data to predict future game spreads.

### Notebooks and Scripts

- `model/predict/nba-predict-spread.ipynb`: Notebook for model development and training.
- `model/predict/nba-predict-v2.ipynb`: Updated model training notebook.

## Streamlit Web App

The web app is built using Streamlit, providing an interactive interface for users to input game details and view predictions.

### Application Files

- `app/app.py`: Main application file.
- `app/predict_page.py`: Script for the prediction page.

## Future Improvements

- **Model Improvement:** Enhance the model accuracy by incorporating advanced machine learning techniques.
- **Data Visualization:** Add more visualizations to the web app to better represent the data and predictions.
- **User Authentication:** Implement user authentication for personalized predictions and history tracking.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.