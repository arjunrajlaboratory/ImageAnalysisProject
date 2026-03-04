# Test File Creation

A test/demo worker that creates a sample CSV file and uploads it to the dataset folder. Used for verifying that file creation and upload to Girder works correctly.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| File name | text | `test_output.csv` | The name of the CSV file to create and upload to the dataset folder. |

## Computed Properties

This worker does not compute annotation properties. It creates a sample CSV file with the following structure:

| Column | Content |
|--------|---------|
| A | Integers 1 through 10 |
| B | Integers 10 through 1 (descending) |
| C | The string "test" repeated 10 times |

## How It Works

1. Creates a hardcoded pandas DataFrame with three columns (A, B, C) and 10 rows of sample data.
2. Converts the DataFrame to CSV format in memory using a StringIO buffer.
3. Retrieves the dataset folder from Girder.
4. Uploads the CSV content to the dataset folder with the user-specified file name and `text/csv` MIME type.

## Notes

- This worker does not read any annotations or image data. It exists purely to test the file upload pipeline.
- Progress is reported at 50% (creating file), 75% (uploading), and 100% (finished).
- The uploaded file will appear in the dataset's Girder folder and can be downloaded from there.
