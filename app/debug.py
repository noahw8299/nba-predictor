import pickle

try:
    with open('predict_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading the model: {e}")

try:
    with open('model_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Error loading the scaler: {e}")
