import os
import logging
import pandas as pd
from config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Directory containing your processed Excel files
PROCESSED_DIR = "./data/processed"
# Output CSV file that will be used for training
TRAINING_DATA_FILE = "./data/training_data.csv"

def add_forecast_column(df):
    """
    Create a new 'forecast' column:
    Set to 1 if any of the max_percent columns > 50, else 0.
    """
    required_cols = ["max_percent_1h", "max_percent_4h", "max_percent_8h", "max_percent_24h"]
    if not all(col in df.columns for col in required_cols):
        logger.error("DataFrame is missing one or more required max_percent columns.")
        return df

    condition = (
            (df["max_percent_1h"] > 50) |
            (df["max_percent_4h"] > 50) |
            (df["max_percent_8h"] > 50) |
            (df["max_percent_24h"] > 50)
    )
    df["forecast"] = condition.astype(int)
    return df

def process_processed_files():
    """
    Iterate over all processed Excel files, add the forecast column,
    and combine them into a single DataFrame.
    """
    combined_df = pd.DataFrame()
    for filename in os.listdir(PROCESSED_DIR):
        if filename.endswith((".xlsx", ".xls")):
            file_path = os.path.join(PROCESSED_DIR, filename)
            logger.info(f"Processing file: {file_path}")
            try:
                df = pd.read_excel(file_path)
                # Ensure required columns are present before adding forecast
                if all(col in df.columns for col in ["max_percent_1h", "max_percent_4h", "max_percent_8h", "max_percent_24h"]):
                    df = add_forecast_column(df)
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                else:
                    logger.warning(f"Skipping {file_path}: missing one or more max_percent columns.")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
    return combined_df

def export_training_data(df):
    """
    Export the combined DataFrame to a CSV file.
    """
    df.to_csv(TRAINING_DATA_FILE, index=False)
    logger.info(f"Exported training data to {TRAINING_DATA_FILE}")

# Optional: Function to trigger Google AutoML training
def train_automl_model():
    """
    Example function to create a dataset in Google AutoML,
    import the training CSV data from a GCS bucket, and trigger a training job.

    Ensure that your config contains:
      - google_automl.project_id
      - google_automl.compute_region (e.g. "us-central1")
      - google_automl.gcs_input_uri (the GCS URI where your CSV file is stored)

    Also, make sure you have the 'google-cloud-automl' library installed.
    """
    from google.cloud import automl_v1beta1 as automl

    config = load_config()
    project_id = config.google_automl.project_id
    compute_region = config.google_automl.compute_region
    gcs_input_uri = config.google_automl.gcs_input_uri  # e.g., "gs://your-bucket/path/to/training_data.csv"
    dataset_display_name = "crypto_forecast_dataset"
    model_display_name = "crypto_forecast_model"

    client = automl.AutoMlClient()
    project_location = client.location_path(project_id, compute_region)

    # Create a new dataset for tables (AutoML Tables)
    dataset = {
        "display_name": dataset_display_name,
        "tables_dataset_metadata": {}  # Additional dataset options can be set here.
    }
    response = client.create_dataset(project_location, dataset)
    dataset_full_id = response.name
    logger.info(f"Created dataset: {dataset_full_id}")

    # Configure input for importing CSV data from GCS
    input_config = {
        "gcs_source": {
            "input_uris": [gcs_input_uri]
        }
    }
    import_response = client.import_data(dataset_full_id, input_config)
    logger.info("Importing data into AutoML dataset...")
    import_response.result()  # Wait for the import to complete

    # Create and train a model (this may take a long time to complete)
    model = {
        "display_name": model_display_name,
        "tables_model_metadata": {}  # Optionally specify model metadata.
    }
    model_response = client.create_model(project_location, dataset_full_id, model)
    logger.info("Training model... (this operation may take several hours)")
    model_response.result()  # Wait for model training to complete
    logger.info(f"Model training complete: {model_response.name}")

def main():
    # Load configuration (includes API keys, GCS URIs, etc.)
    config = load_config()

    # Process all processed Excel files and add forecast column
    training_df = process_processed_files()
    if training_df.empty:
        logger.error("No valid training data found. Exiting.")
        return

    # Export the combined DataFrame to a CSV file (this file can be uploaded to GCS if needed)
    export_training_data(training_df)

    # Optionally, if you have set up your GCS bucket and AutoML credentials,
    # upload the CSV file to GCS and then trigger training.
    # For example:
    # upload_file_to_gcs(TRAINING_DATA_FILE, config.google_automl.gcs_bucket)
    # train_automl_model()

if __name__ == "__main__":
    main()
