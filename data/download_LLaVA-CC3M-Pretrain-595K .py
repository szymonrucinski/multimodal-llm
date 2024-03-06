import requests
import logging

logger = logging.getLogger(__name__)
# The URL of the file you want to download
url = "https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/resolve/main/chat.json"

# The local path where you want to save the file
output_filename = "chat.json"

# Send a GET request to the URL
response = requests.get(url, allow_redirects=True)

# Check if the request was successful
if response.status_code == 200:
    # Write the content of the response to a file
    with open(output_filename, "wb") as file:
        file.write(response.content)
    logger.debug(f"File downloaded successfully and saved as {output_filename}")
else:
    logger.error(f"Failed to download the file. Status code: {response.status_code}")
