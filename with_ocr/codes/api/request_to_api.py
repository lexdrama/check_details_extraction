import requests

# Replace 'http://localhost:8005' with the actual URL where your FastAPI server is running
api_url = 'http://localhost:8005/process_check'

# Replace 'path/to/your/image.jpg' with the actual path to your image file
image_path = "with_ocr/codes/post_train/1.JPG"

# Append the image_path to the URL
#api_url_with_path = f'{api_url}?image_path={image_path}'

# Append the image_path to the URL
api_url_with_path = f'{api_url}/{image_path}'

# Prepare the payload as a dictionary with the image_path parameter
#payload = {'image_path': image_path}

# Make a POST request to the API
response = requests.post(api_url_with_path)

# Check the response status code
if response.status_code == 200:
    # Print the JSON response
    print(response.json())
else:
    # Print an error message if the request was not successful
    print(f"Error: {response.status_code}, {response.text}")
