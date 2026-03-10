import json

import requests

if __name__ == "__main__":
    # Define YOUR OWN authservice_session for authentication
    # cookies = {
    #     "authservice_session": "MTc0MjgwNzkwM3xOd3dBTkRWWVNGUTNRbE5NU0ZWSk5GWTBUemRPVWtKRlQwMU9XVkZOTlVJMlNEZERORUZTV1VwVE5USXlUVFZEV0RkSU1rdFdVMUU9fDyVC2naRvabzMYWWoUzTu3sHYUxxCBZwG7v5cFQEJIE",
    # }

    # We will send requests with content-type is json
    headers = {
        "Host": "diabete-classifier.eventing-test.example.com",
        # "Host": "intrusion-detection.kserve-test.example.com",
        "Content-Type": "application/json"
    }

    # Define our data for prediction
    data = {
    "input_data": [
        [6,103,66,0,0,24.3,0.249,29],
        [3,89,74,16,85,30.4,0.551,38]
    ]
    }

    response = requests.post(
        "http://localhost:8081/v1/models/diabetest-model:predict",
        # cookies=cookies,
        data=json.dumps(data),
        headers=headers,
    )

    print(response.json())