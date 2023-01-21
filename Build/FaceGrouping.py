# Define an array of face IDs
face_ids = ["face1_id", "face2_id", "face3_id"]

# Send a POST request to the Face - Group API
response = requests.post(
    endpoint + "/group",
    headers=headers,
    json={"faceIds": face_ids}
)

# Parse the JSON response
response_json = json.loads(response.text)

# Extract the groups of similar faces
groups = response_json["groups"]
