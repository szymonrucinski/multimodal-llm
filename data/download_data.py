import requests
import logging
import json

logger = logging.getLogger(__name__)
# The URL of the file you want to download
import zipfile


def unzip_if_zip(filename):
    # Check if filename ends with '.zip'
    if filename.endswith(".zip"):
        # Attempt to unzip the file
        try:
            with zipfile.ZipFile(filename, "r") as zip_ref:
                # Extract all the contents into the directory
                extract_dir = filename.replace(
                    ".zip", ""
                )  # Name the extraction directory
                zip_ref.extractall(extract_dir)
                print(f"File '{filename}' successfully unzipped to '{extract_dir}'")
        except zipfile.BadZipFile:
            print(f"Error: The file '{filename}' is not a valid zip file.")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print(f"The file '{filename}' is not a zip file.")


# Example usage
# Load the list of dictionaries from the JSON file
with open("file_list.json", "r") as file:
    file_list = json.load(file)


for f in file_list:
    # Send a GET request to the URL
    response = requests.get(f["url"], allow_redirects=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the content of the response to a file
        with open(f["output_filename"], "wb") as file:
            file.write(response.content)
        logger.debug(
            f"File downloaded successfully and saved as {f['output_filename']}"
        )
    else:
        logger.error(
            f"Failed to download the file. Status code: {response.status_code}"
        )
