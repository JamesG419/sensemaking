from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from sensemaking.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "activity-log-TOTO.csv",
    fact_path: Path = PROCESSED_DATA_DIR / "activity_log_facts.csv",
    user_path: Path = PROCESSED_DATA_DIR / "activity_log_users.csv",
    document_path: Path = PROCESSED_DATA_DIR / "activity_log_documents.csv",
    image_path: Path = PROCESSED_DATA_DIR / "activity_log_images.csv",
    # ----------------------------------------------
):
    #
    logger.info("Processing dataset...")
    # Read the input CSV file
    df = pd.read_csv(input_path)
    logger.info(f"Read {len(df)} rows from {input_path}")
    # Process the data and save to different files
    logger.info("Processing users...")
    user_dim = (
        df[["User ID", "Username", "Team"]]
        .drop_duplicates()
        .rename(columns={"User ID": "user_id", "Username": "username", "Team": "team"})
        .reset_index(drop=True)
    )

    logger.info(f"Found {len(user_dim)} unique users")
    logger.info("Processing documents...")
    document_dim = (
        df[["Document ID", "Line ID", "Line Number", "Original Text"]]
        .drop_duplicates()
        .rename(
            columns={
                "Document ID": "document_id",
                "Line ID": "line_id",
                "Line Number": "line_number",
                "Original Text": "original_text",
            }
        )
        .reset_index(drop=True)
    )
    logger.info(f"Found {len(document_dim)} unique documents")
    # Add surrogate key
    document_dim["document_line_id"] = document_dim.index + 1

    df = df.merge(
        document_dim,
        how="left",
        left_on=["Document ID", "Line ID", "Line Number", "Original Text"],
        right_on=["document_id", "line_id", "line_number", "original_text"],
    )
    document_dim = document_dim[
        ["document_line_id", "document_id", "line_id", "line_number", "original_text"]
    ]

    logger.info("Processing images...")
    image_dim = (
        df[["Image ID", "Image Name", "Image Notes"]]
        .dropna(subset=["Image ID"])
        .drop_duplicates()
        .drop_duplicates()
        .rename(
            columns={
                "Image ID": "image_id",
                "Image Name": "image_name",
                "Image Notes": "image_notes",
            }
        )
        .reset_index(drop=True)
    )
    logger.info(f"Found {len(image_dim)} unique images")

    # Create facts table
    logger.info("Creating facts table...")
    activity_log_facts = (
        df[
            [
                "Activity Type",
                "Date",
                "Time",
                "User ID",
                "document_line_id",
                "Image ID",
                "New Text",
            ]
        ]
        .rename(
            columns={
                "Activity Type": "activity_type",
                "Date": "date",
                "Time": "time",
                "User ID": "user_id",
                "Image ID": "image_id",
                "New Text": "new_text",
            }
        )
        .reset_index(drop=True)
    )

    activity_log_facts["activity_id"] = activity_log_facts.index + 1
    activity_log_facts = activity_log_facts[
        [
            "activity_id",
            "activity_type",
            "date",
            "time",
            "user_id",
            "document_line_id",
            "image_id",
            "new_text",
        ]
    ]

    # Save the processed data to CSV files
    logger.info(f"Saving data to {fact_path}, {user_path}, {document_path}, {image_path}")
    activity_log_facts.to_csv(fact_path, index=False)
    user_dim.to_csv(user_path, index=False)
    document_dim.to_csv(document_path, index=False)
    image_dim.to_csv(image_path, index=False)

    logger.info("Processing complete!")


if __name__ == "__main__":
    app()
