import json
print("\n--- cURL Example ---")
instruction = "Stack the black cube on top of the red cube"
objects = [
    {"red-cube": [51, 43, 17]},
    {"black-cube": [44, 58, 17]},
    {"purple-cube": [74, 59, 17]},
    {"green-cube": [65, 82, 17]},
]
curl_command = (
    "curl -X POST "
    "-H \"Content-Type: application/json\" "
    "-d '"
    + json.dumps({
        "instruction": instruction,
        "objects": objects
    })
    + "' "
    + f"http://localhost:8000/robot/task" # Correct base URL usage

)
print(curl_command)