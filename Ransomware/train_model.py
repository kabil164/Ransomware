import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Define features
features = ['Machine', 'DebugSize', 'MajorImageVersion', 'ExportSize',
            'IatVRA', 'NumberOfSections', 'SizeOfStackReserve',
            'DllCharacteristics', 'ResourceSize', 'BitcoinAddresses']

def load_data(file_path='ransomware_data.csv'):
    """
    Load dataset (placeholder: replace with actual data source).
    Expected: CSV with 10 features and 'label' column (1=benign, 0=malicious).
    """
    try:
        df = pd.read_csv(file_path)
        if not all(f in df.columns for f in features + ['label']):
            raise ValueError("Dataset missing required columns.")
        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please provide a valid dataset.")
        return None

def preprocess_data(df):
    """Preprocess data: handle missing values, ensure numeric types."""
    df = df.copy()
    df[features] = df[features].fillna(0)  # Replace NaN with 0 (adjust as needed)
    df[features] = df[features].astype(int)  # Ensure integer types
    return df

def train_model():
    """Train Random Forest model and save it."""
    df = load_data()
    if df is None:
        return

    # Preprocess
    df = preprocess_data(df)
    X = df[features]
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Malicious', 'Benign']))

    # Save model
    joblib.dump(model, 'best_model.pkl')
    print("Model saved as 'best_model.pkl'.")

if __name__ == '__main__':
    train_model()