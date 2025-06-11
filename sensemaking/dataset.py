from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from sensemaking.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def find_chars(test_string):
    string_output = {"empty": False}  # something not there
    if isinstance(test_string, float) and pd.isna(test_string):
        return {"empty": True}
    if not (test_string):  # nothing there
        return {"empty": True}
    string_output["@"] = test_string.count("@")
    string_output[">"] = test_string.count(">")
    string_output["+"] = test_string.count("+")
    string_output["&"] = test_string.count("&")
    string_output["~"] = test_string.count("~")
    string_output["NEW"] = test_string.count("NEW") or test_string.count("new")
    return string_output


def categorize_cell(data):
    output = ""
    if data["empty"]:
        return "Empty"
    if all(not x for x in data.values()):
        return "No Links"
    if data["NEW"] > 0:
        output = f"NEW{output}"
    if data["@"] > 0 and data["@"] <= 1:
        output = f"{output} Account"
    if data["@"] > 1:
        output = f"{output} Account Account"
    if data[">"] > 0:
        output = f"{output} BEND"
    if data["+"] > 0 and data["+"] <= 1:
        output = f"{output} Tag"
    if data["+"] > 1:
        output = f"{output} Tag Tag"
    if data["&"] > 0 and data["&"] <= 1:
        output = f"{output} Group"
    if data["&"] > 1:
        output = f"{output} Group Group"
    if data["~"] > 0:
        output = f"{output} Class"

    return output


def cell_description(data):
    cell = [
        (data == "Empty"),
        (data == "No Links"),
        (data == "NEW"),
        (data == "NEW Account"),
        (data == "NEW Account Account"),
        (data == "NEW Account Group"),
        (data == "NEW Account Account Group"),
        (data == "NEW Account Tag"),
        (data == "NEW Account Account Tag"),
        (data == "NEW Account BEND"),
        (data == "NEW Account Account BEND"),
        (data == "NEW Account Class"),
        (data == "NEW Account Account Class"),
        (data == "NEW Group"),
        (data == "NEW Group Group"),
        (data == "NEW Tag"),
        (data == "NEW Tag Tag"),
        (data == "NEW Tag Group"),
        (data == "NEW Tag Group Group"),
        (data == "NEW Tag Tag Group"),
        (data == " Account"),
        (data == " Account Account"),
        (data == " Account BEND"),
        (data == " Account Account BEND"),
        (data == " Account Tag"),
        (data == " Account Account Tag"),
        (data == " Account Tag Tag"),
        (data == " Account Account Tag Tag"),
        (data == " Account BEND Tag"),
        (data == " Account Account BEND Tag"),
        (data == " Account BEND Tag Tag"),
        (data == " Account Account BEND Tag Tag"),
        (data == " Account Group"),
        (data == " Account Account Group"),
        (data == " Account Group Group"),
        (data == " Account Account Group Group"),
        (data == " Account BEND Group"),
        (data == " Account Account BEND Group"),
        (data == " Account BEND Group Group"),
        (data == " Account Account BEND Group Group"),
        (data == " Account Tag Group"),
        (data == " Account Account Tag Group"),
        (data == " Account Account Tag Tag Group"),
        (data == " Account Account Tag Tag Group Group"),
        (data == " Account BEND Tag Group"),
        (data == " Account Account BEND Tag Group"),
        (data == " Account BEND Tag Tag Group"),
        (data == " Account Account BEND Tag Tag Group"),
        (data == " Account BEND Tag Group Group"),
        (data == " Account Account BEND Tag Group Group"),
        (data == " Account BEND Tag Tag Group Group"),
        (data == " Account Account BEND Tag Tag Group Group"),
        (data == " Account Class"),
        (data == " Account Account Class"),
        (data == " Account BEND Class"),
        (data == " Account Account BEND Class"),
        (data == " Account Tag Class"),
        (data == " Account Account Tag Class"),
        (data == " Account Account Tag Tag Class"),
        (data == " Account Group Class"),
        (data == " Account Account Group Class"),
        (data == " Account Account Group Group Class"),
        (data == " Account BEND Tag Class"),
        (data == " Account BEND Tag Tag Class"),
        (data == " Account Account BEND Tag Class"),
        (data == " Account Account BEND Tag Tag Class"),
        (data == " Account BEND Tag Group Class"),
        (data == " Account Account BEND Tag Group Class"),
        (data == " Account BEND Tag Tag Group Class"),
        (data == " Account Account BEND Tag Tag Group Class"),
        (data == " Account BEND Tag Group Group Class"),
        (data == " Account Account BEND Tag Group Group Class"),
        (data == " Account BEND Tag Tag Group Group Class"),
        (data == " Account Account BEND Tag Tag Group Group Class"),
        (data == " BEND"),
        (data == " BEND Tag"),
        (data == " BEND Tag Tag"),
        (data == " BEND Group"),
        (data == " BEND Group Group"),
        (data == " BEND Tag Group"),
        (data == " BEND Tag Tag Group"),
        (data == " BEND Tag Tag Group Group"),
        (data == " BEND Tag Group Group"),
        (data == " BEND Class"),
        (data == " BEND Tag Class"),
        (data == " BEND Tag Tag Class"),
        (data == " BEND Tag Group Class"),
        (data == " BEND Tag Tag Group Class"),
        (data == " BEND Tag Group Group Class"),
        (data == " BEND Tag Tag Group Group Class"),
        (data == " Tag"),
        (data == " Tag Tag"),
        (data == " Tag Group"),
        (data == " Tag Tag Group"),
        (data == " Tag Tag Group Group"),
        (data == " Tag Class"),
        (data == " Tag Tag Class"),
        (data == " Tag Group Class"),
        (data == " Tag Tag Group Class"),
        (data == " Tag Group Group Class"),
        (data == " Tag Tag Group Group Class"),
        (data == " Group"),
        (data == " Group Group"),
        (data == " Group Class"),
        (data == " Group Group Class"),
        (data == " Class"),
    ]

    description = [
        "Empty",
        "No Linkages",
        "No Linkages",
        "New Account",
        "New Account",
        "Linkage - Account and Group",
        "Linkage - Account and Group",
        "Linkage - Account and Tag",
        "Linkage - Account and Tag",
        "Linkage - Account and BEND",
        "Linkage - Account and BEND",
        "Account Classification",
        "Account Classification",
        "New Group",
        "New Group",
        "New Tag",
        "New Tag",
        "Linkage - Tag and Group",
        "Linkage - Tag and Group",
        "Linkage - Tag and Group",
        "Account Statement",
        "Linkage - Account and Account",
        "Linkage - Account and BEND",
        "Linkage - Account and BEND",
        "Linkage - Account and Tag",
        "Linkage - Account and Tag",
        "Linkage - Account and Tag",
        "Linkage - Account and Tag",
        "Linkage - Account, BEND, Tag",
        "Linkage - Account, BEND, Tag",
        "Linkage - Account, BEND, Tag",
        "Linkage - Account, BEND, Tag",
        "Linkage - Account and Group",
        "Linkage - Account and Group",
        "Linkage - Account and Group",
        "Linkage - Account and Group",
        "Linkage - Account, BEND, Group",
        "Linkage - Account, BEND, Group",
        "Linkage - Account, BEND, Group",
        "Linkage - Account, BEND, Group",
        "Linkage - Account, Tag, Group",
        "Linkage - Account, Tag, Group",
        "Linkage - Account, Tag, Group",
        "Linkage - Account, Tag, Group",
        "Linkage - Account, BEND, Tag, Group",
        "Linkage - Account, BEND, Tag, Group",
        "Linkage - Account, BEND, Tag, Group",
        "Linkage - Account, BEND, Tag, Group",
        "Linkage - Account, BEND, Tag, Group",
        "Linkage - Account, BEND, Tag, Group",
        "Linkage - Account, BEND, Tag, Group",
        "Linkage - Account, BEND, Tag, Group",
        "Account Classification",
        "Account Classification",
        "Linkage - Account, BEND, Account Classification",
        "Linkage - Account, BEND, Account Classification",
        "Linkage - Account, Tag, Account Classifcation",
        "Linkage - Account, Tag, Account Classifcation",
        "Linkage - Account, Tag, Account Classifcation",
        "Linkage - Account, Group, Account Classification",
        "Linkage - Account, Group, Account Classification",
        "Linkage - Account, Group, Account Classification",
        "Linkage - Account, BEND, Tag, Account Classification",
        "Linkage - Account, BEND, Tag, Account Classification",
        "Linkage - Account, BEND, Tag, Account Classification",
        "Linkage - Account, BEND, Tag, Account Classification",
        "Linkage - Account, BEND, Tag, Group, Account Classification",
        "Linkage - Account, BEND, Tag, Group, Account Classification",
        "Linkage - Account, BEND, Tag, Group, Account Classification",
        "Linkage - Account, BEND, Tag, Group, Account Classification",
        "Linkage - Account, BEND, Tag, Group, Account Classification",
        "Linkage - Account, BEND, Tag, Group, Account Classification",
        "Linkage - Account, BEND, Tag, Group, Account Classification",
        "Linkage - Account, BEND, Tag, Group, Account Classification",
        "BEND Statement",
        "Linkage - BEND and Tag",
        "Linkage - BEND and Tag",
        "Linkage - BEND and Group",
        "Linkage - BEND and Group",
        "Linkage - BEND, Tag, Group",
        "Linkage - BEND, Tag, Group",
        "Linkage - BEND, Tag, Group",
        "Linkage - BEND, Tag, Group",
        "Linkage - BEND and Account Classification",
        "Linkage - BEND Tag and Account Classification",
        "Linkage - BEND Tag and Account Classification",
        "Linkage - BEND, Tag, Group, Account Classification",
        "Linkage - BEND, Tag, Group, Account Classification",
        "Linkage - BEND, Tag, Group, Account Classification",
        "Linkage - BEND, Tag, Group, Account Classification",
        "Tag Statement",
        "Tag Statement",
        "Linkage - Tag and Group",
        "Linkage - Tag and Group",
        "Linkage - Tag and Group",
        "Linkage - Tag and Account Classification",
        "Linkage - Tag and Account Classification",
        "Linkage - Tag, Group, Account Classification",
        "Linkage - Tag, Group, Account Classification",
        "Linkage - Tag, Group, Account Classification",
        "Linkage - Tag, Group, Account Classification",
        "Group Statement",
        "Group Statement",
        "Linkage - Group and Account Classification",
        "Linkage - Group and Account Classification",
        "Account Classification Statement",
    ]

    return np.select(cell, description, "")


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "activity-log-TOTO.csv",
    fact_path: Path = PROCESSED_DATA_DIR / "activity_log_facts.csv",
    user_path: Path = PROCESSED_DATA_DIR / "activity_log_users.csv",
    document_path: Path = PROCESSED_DATA_DIR / "activity_log_documents.csv",
    image_path: Path = PROCESSED_DATA_DIR / "activity_log_images.csv",
    action_path: Path = PROCESSED_DATA_DIR / "activity_log_actions.csv",
    # ----------------------------------------------
):
    #
    logger.info("Processing dataset...")
    # Read the input CSV file
    df = pd.read_csv(input_path)
    # Create Type of Action column
    df["Type of Action"] = df["New Text"].apply(
        lambda x: cell_description(categorize_cell(find_chars(x)))
    )
    df["Type of Action"] = np.where(
        (df["Activity Type"] == "set_image") & (df["Image Name"] == "null"),
        "New Image",
        df["Type of Action"],
    )
    df["Activity Type"] = np.where(
        df["Type of Action"] == "New Image", "Add Image", df["Activity Type"]
    )

    # Edited images
    df["Type of Action"] = np.where(
        (df["Activity Type"] == "set_image"), "Edit Image", df["Type of Action"]
    )
    df["Activity Type"] = np.where(
        df["Type of Action"] == "Edit Image", "Edit Image", df["Activity Type"]
    )

    # Line Deletion
    df["Type of Action"] = np.where(
        (df["Type of Action"] == "Empty") & (df["Activity Type"] == "delete"),
        "Line Deletion",
        df["Type of Action"],
    )
    df["Type of Action"] = np.where(
        (df["Type of Action"] == "Empty") & (df["Activity Type"] == "edit"),
        "Line Deletion",
        df["Type of Action"],
    )

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

    # Create actions table
    logger.info("Creating actions table...")
    action_dim = (
        df[["Type of Action"]]
        .drop_duplicates()
        .rename(columns={"Type of Action": "category_label"})
        .reset_index(drop=True)
    )

    action_dim["action_type_id"] = action_dim.index + 1

    action_dim = action_dim[["action_type_id", "category_label"]]

    logger.info(f"Found {len(action_dim)} unique actions")

    df = df.merge(
        action_dim,
        how="left",
        left_on="Type of Action",
        right_on="category_label",
    )

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
                "action_type_id",
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
            "action_type_id",
            "new_text",
        ]
    ]

    # Save the processed data to CSV files
    logger.info(f"Saving data to {fact_path}, {user_path}, {document_path}, {image_path}")
    activity_log_facts.to_csv(fact_path, index=False)
    user_dim.to_csv(user_path, index=False)
    document_dim.to_csv(document_path, index=False)
    image_dim.to_csv(image_path, index=False)
    action_dim.to_csv(action_path, index=False)

    logger.info("Processing complete!")


if __name__ == "__main__":
    app()
