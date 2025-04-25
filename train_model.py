import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import joblib
import os # To ensure directory exists

# --- 1. Data Simulation ---
def simulate_campaign_data(num_samples=1000):
    """Generates synthetic marketing campaign data."""
    np.random.seed(42) # for reproducibility

    channels = ['Facebook', 'Google Ads', 'LinkedIn', 'TikTok', 'Email']
    ad_types = ['Image', 'Video', 'Carousel', 'Text']
    ctas = ['Shop Now', 'Learn More', 'Sign Up', 'Download', 'Contact Us']
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    data = {
        'Channel': np.random.choice(channels, num_samples),
        'BudgetTargetAudienceSize': np.random.randint(10000, 500000, num_samples), # Example: combined metric or proxy
        'AdType': np.random.choice(ad_types, num_samples),
        'CallToAction': np.random.choice(ctas, num_samples),
        'DayOfWeek': np.random.choice(days, num_samples),
        'Budget': np.random.uniform(100, 5000, num_samples) # Adding budget as a separate feature
    }
    df = pd.DataFrame(data)

    # Simulate CTR (Target Variable) - Make it logically depend *slightly* on features
    base_ctr = np.random.normal(loc=2.0, scale=0.5, size=num_samples) # Base CTR around 2%

    # Add some feature influence (example logic)
    ctr_influence = (
        + (df['Channel'] == 'Facebook').astype(int) * 0.3
        + (df['Channel'] == 'Google Ads').astype(int) * 0.4
        + (df['AdType'] == 'Video').astype(int) * 0.5
        + (df['CallToAction'] == 'Shop Now').astype(int) * 0.2
        - (df['BudgetTargetAudienceSize'] / 500000) * 0.5 # Larger audience slightly lower CTR?
        + (df['Budget'] / 5000) * 0.3 # Higher budget slightly higher CTR?
        + np.random.normal(0, 0.3, num_samples) # Add more noise
    )
    df['CTR'] = np.clip(base_ctr + ctr_influence, 0.2, 10.0) # Ensure CTR is within reasonable bounds (0.2% to 10%)

    return df

# --- 2. Preprocessing & Training ---
def train_model(df):
    """Preprocesses data, trains a model, and saves artifacts."""
    print("Starting model training...")

    X = df.drop('CTR', axis=1)
    y = df['CTR']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify categorical and numerical features
    categorical_features = ['Channel', 'AdType', 'CallToAction', 'DayOfWeek']
    numerical_features = ['BudgetTargetAudienceSize', 'Budget']

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # Use sparse_output=False for easier handling later

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any (none in this case)
    )

    # Create the full pipeline including preprocessing and the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)) # n_jobs=-1 uses all processors
    ])

    # Train the model
    print("Fitting pipeline...")
    model_pipeline.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model Mean Absolute Error (MAE) on Test Set: {mae:.4f}% CTR")

    # --- 3. Save Artifacts ---
    # Ensure the 'artifacts' directory exists
    artifacts_dir = 'artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)

    print("Saving model pipeline...")
    pipeline_path = os.path.join(artifacts_dir, 'model_pipeline.joblib')
    joblib.dump(model_pipeline, pipeline_path)
    print(f"Model pipeline saved to {pipeline_path}")

    # (Optional but good practice) Save column names used during training for the app
    # This helps ensure the app sends data in the correct format/order
    columns_path = os.path.join(artifacts_dir, 'training_columns.joblib')
    training_columns = list(X_train.columns)
    joblib.dump(training_columns, columns_path)
    print(f"Training columns saved to {columns_path}")

    print("Training complete.")
    return pipeline_path # Return path for confirmation

# --- Main Execution ---
if __name__ == "__main__":
    print("Simulating data...")
    campaign_df = simulate_campaign_data(num_samples=2000) # Increase samples for potentially better model
    print(f"Generated {len(campaign_df)} samples.")
    print(campaign_df.head())
    print("\nData Types:\n", campaign_df.dtypes)

    train_model(campaign_df)