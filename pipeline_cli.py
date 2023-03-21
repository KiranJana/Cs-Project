# Spacy_transformers/pipeline_cli.py

import click
import pandas as pd
import gcsfs

from training_data import TrainingData

# CLI
# define commands group
@click.group()
def main():
    pass


# define command pre_label
@main.command()
@click.argument("scraped_blob_name", type=str)
@click.argument("bucket_name", type=str)
@click.argument("rules_blob_name", type=str)
@click.argument("save_path", type=str)
@click.option(
    "--google_credentials_path",
    "-credentials",
    type=click.STRING,
    help="Provide path to Google service account key, if none, environment variable pointing is used...",
)
def pre_label(
    bucket_name, rules_blob_name, scraped_blob_name, save_path, google_credentials_path
):
    """
    Pre-label scraped text via spacy and save as json.\n
    arguments\n
    : scraped_bob_name: name of a blob consisting the scraped data in GCS (data scraped by indded-scraper)\n
    : bucket_name: name of a bucket holding the scraped data and the rules JSON file (must be common for both)\n
    : rules_blob_name: name of a blob with spacy entity ruler rules (used for prelabelling)\n
    : save_path: local or GCS path to store the labeled data (path to GCS must be with gs://...)
    options\n
    : google_credentials_path: path to json service account key
    """
    click.echo(
        f"scraped_blob_name={scraped_blob_name}, bucket_name={bucket_name}, rules_blob_name={rules_blob_name}, save_path={save_path}"
    )

    source = f"gs://{bucket_name}/{scraped_blob_name}"
    try:
        fs = gcsfs.GCSFileSystem(token=google_credentials_path)
        with fs.open(source) as f:
            scraped_data = pd.read_parquet(f)
    except:
        click.echo(
            "Google credentials not provided or invalid, trying to read from environment variable..."
        )
        scraped_data = pd.read_parquet(f"gs://{bucket_name}/{scraped_blob_name}")
    raw_texts = scraped_data["text"].to_list()

    training_data = TrainingData.pre_label_data(raw_texts, bucket_name, rules_blob_name)

    # save training data to GCS
    training_data.save_as_json(save_path)

    # report outcome
    click.echo(f"Data saved to location {save_path}")


# define command load data and upload to labelbox
@main.command()
@click.argument("load_path", type=str)
@click.argument("labelbox_project_name", type=str)
@click.option(
    "--labelbox_dataset_name",
    "-dataset_name",
    type=click.STRING,
    help="Provide labelbox dataset name to be used, if not provided, project name + dataset is used",
)
@click.option(
    "--labelbox_api_key",
    "-api_key",
    type=click.STRING,
    help="Labelboxa API key, if not provided env var LABELBOX_API_KEY is looked-up",
)
def load_data_and_upload_to_labelbox(
    load_path, labelbox_project_name, labelbox_dataset_name, labelbox_api_key
):
    """
    Load training data from json (local or GCS) and upload to labelbox for manual labeling.
    """
    click.echo(
        f"Params: load_path={load_path}, labelbox_project_name={labelbox_project_name}, labelbox_dataset_name={labelbox_dataset_name}, labebox_api_key={labelbox_api_key}"
    )
    training_data = TrainingData.load_from_json(load_path)
    training_data.upload_data_to_labelbox(
        labelbox_project_name, labelbox_dataset_name, labelbox_api_key
    )
    click.echo(f"Project {labelbox_project_name} uploaded to Labelbox")


if __name__ == "__main__":
    main()
