"""
One-time script to upload the CA weather/fire CSV to BigQuery.

Usage:
    pip install google-cloud-bigquery pandas pyarrow
    python scripts/upload_to_bigquery.py
"""

import pathlib
import pandas as pd
from google.cloud import bigquery

PROJECT = 'sjsu-ds-projects'
DATASET = 'wildfire'
TABLE   = 'ca_weather_fire'

CSV_PATH = pathlib.Path(__file__).parent.parent / 'data' / \
    'CA_Weather_Fire_Dataset_1984-2025-WeatherConditionsDaysOfFire.csv'

def main():
    df = pd.read_csv(CSV_PATH)
    df['FIRE_START_DAY'] = df['FIRE_START_DAY'].astype(str).str.strip().map(
        {'True': True, 'False': False}
    )

    client = bigquery.Client(project=PROJECT)

    dataset_ref = bigquery.Dataset(f'{PROJECT}.{DATASET}')
    dataset_ref.location = 'US'
    client.create_dataset(dataset_ref, exists_ok=True)
    print(f'Dataset {PROJECT}.{DATASET} ready.')

    table_ref  = f'{PROJECT}.{DATASET}.{TABLE}'
    job_config = bigquery.LoadJobConfig(
        write_disposition='WRITE_TRUNCATE',
        autodetect=True,
    )
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    print(f'Loaded {job.output_rows} rows into {table_ref}')

if __name__ == '__main__':
    main()
