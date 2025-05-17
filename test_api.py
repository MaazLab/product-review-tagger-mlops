import requests

# API endpoint
url = "http://localhost:8000/predict"

# Review input
payload = {
    "review_text": "The product quality is excellent but the delivery was delayed and packaging was torn."
}

# Send POST request
response = requests.post(url, json=payload)

# Print result
if response.status_code == 200:
    result = response.json()
    print("Input Review:", result['input'])
    print("Predicted Tags:", result['predicted_tags'])
else:
    print("Error:", response.status_code, response.text)
