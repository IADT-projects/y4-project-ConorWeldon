import requests
import json
import cv2
import numpy as np

# Replace <subscription_key> with your Azure Face API subscription key
subscription_key = "bb67bd956c014f3ea8cf89621c75de21"

# Replace <endpoint> with the endpoint for your Azure Face API instance
endpoint = "https://smartemotionalmirror.cognitiveservices.azure.com"

# Define headers for the API request, including the subscription key
headers = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key
}

def recognize_emotion_and_face():
    """
    Recognize emotions and faces in a webcam video stream using Azure Face API

    Returns:
        dict: a dictionary of emotions and faces information or None if an error occurs
    """
    # Open a connection to the default webcam
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Unable to open webcam")
    except Exception as e:
        print("Error: ", e)
        return None

    # Define the parameters for the API request
    params = {
        'returnFaceAttributes': 'headpose,mask,qualityforrecognition',
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'true',
        'recognitionModel': 'recognition_04',
        'returnRecognitionModel': 'true',
        'detectionModel': 'detection_03',
        'faceIdTimeToLive': '86400'
    }

    while True:
        # Capture a frame from the webcam video stream
        ret, frame = cap.read()
        print(ret) # add this line to print the value of ret

        # Display the live camera feed
        cv2.imshow('Live Camera Feed', frame)
        if cv2.waitKey(5000) & 0xFF == ord('q'):
            break

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        image_data = np.array(buffer).tobytes()

        if not ret:
            print("Error: An error occurred while encoding the image.")
            return None

        # Send a POST request to the Azure Face API endpoint
        try:
            response = requests.post(
                endpoint + "/face/v1.0/detect",
                headers=headers,
                params=params,
                data=image_data
            )
            response.raise_for_status()
        # handle http errors
        except requests.exceptions.HTTPError as err:
            print(f"Error: An error occurred while calling the API: {err}")
            return None

        # Parse the JSON response
        try:
            response_json = json.loads(response.text)
        except json.decoder.JSONDecodeError as err:
            print(f"Error: An error occurred while parsing the response: {err}")
            return None

        # check if faces are detected
        if len(response_json) > 0:
            # Convert the first face found to JPEG format and send it to Azure Face API
            try:
                top = response_json[0]['faceRectangle']['top']
                left = response_json[0]['faceRectangle']['left']
                height = response_json[0]['faceRectangle']['height']
                width = response_json[0]['faceRectangle']['width']

                # Access image data as a NumPy array using integer or slice indices
                # Convert the image data to a NumPy array
                img_np = np.frombuffer(image_data, dtype=np.uint8)
                # Decode the image
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                # Crop the face region
                face_data = img[top:top+height, left:left+width, :]
                _, face_jpeg_data = cv2.imencode('.jpg', face_data)
            except Exception as e:
                print(f"Error: An error occurred while cropping the face: {e}")
                return None



            try:
                response = requests.post(
                    endpoint + "/face/v1.0/detect",
                    headers=headers,
                    params=params,
                    data=face_jpeg_data.tobytes()
                )
                response.raise_for_status()
            except requests.exceptions.HTTPError as err:
                print(f"Error: An error occurred while calling the API: {err}")
                return None

            try:
                response_json = json.loads(response.text)
            except json.decoder.JSONDecodeError as err:
                print(f"Error: An error occurred while parsing the response: {err}")
                return None

            # check if emotions are detected
            if len(response_json) > 0 and 'faceAttributes' in response_json[0]:
                emotions = response_json[0]['faceAttributes']['emotion']
                return emotions
            else:
                print("No emotions were detected in the face.")
        else:
            print("No faces were detected in the image.")

        # Stop the webcam capture and close the window after processing the first faceq
        # break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    result = recognize_emotion_and_face()
    print(result)
