import requests

# API endpoint
url = "http://127.0.0.1:8000/upload/"

# File path (replace with your actual CSV file)
file_path = "student_performance_large_dataset.csv"

# Open the file and send it in the request
with open(file_path, "rb") as file:
    files = {"file": file}
    response = requests.post(url, files=files)

# Print the response
print(response.status_code)  # Should be 200 if successful
print(response.json())  # Print the API response as JSON
