import os
import cv2
import time
from datetime import datetime
import joblib
import csv
import numpy as np
import streamlit as st
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import io
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# Function to get the path to the client secret file
def get_client_secret_path():
    return os.path.join(os.path.dirname(__file__), '.streamlit', 'client_secret_930969653781-dpcai4tesnfihldtjbup7oitdrpqij3g.apps.googleusercontent.com.json')

# Authenticate Google Drive API
def authenticate_google_drive():
    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = None
    token_path = 'token.pickle'
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(get_client_secret_path(), SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
    return creds

# Function to fetch files from a specific Google Drive folder
def fetch_files_from_folder(service, folder_id):
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    return results.get('files', [])

# Function to download a file from Google Drive
def download_file(service, file_id, file_name):
    request = service.files().get_media(fileId=file_id)
    file_data = io.BytesIO()
    downloader = MediaIoBaseDownload(file_data, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    file_data.seek(0)
    with open(file_name, 'wb') as f:
        f.write(file_data.read())

# Function to upload a file to Google Drive
def upload_file_to_drive(service, folder_id, file_path, file_name):
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

# Main function to mark attendance
def mark_attendance():
    # Authenticate Google Drive
    creds = authenticate_google_drive()
    service = build('drive', 'v3', credentials=creds)

    # Get the 'Attendance' folder ID
    attendance_folder_name = "Attendance"
    results = service.files().list(
        q=f"name='{attendance_folder_name}' and mimeType='application/vnd.google-apps.folder'",
        fields='files(id)').execute()
    items = results.get('files', [])
    if not items:
        st.error("Attendance folder not found in Google Drive.")
        return
    attendance_folder_id = items[0]['id']

    # Get the 'register' folder ID or create it if not found
    register_folder_id = None
    results = service.files().list(q=f"'{attendance_folder_id}' in parents", fields='files(id, name)').execute()
    for item in results.get('files', []):
        if item['name'] == 'register':
            register_folder_id = item['id']

    if not register_folder_id:
        # Create the 'register' folder
        file_metadata = {
            'name': 'register',
            'parents': [attendance_folder_id],
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        register_folder_id = folder.get('id')
        st.write("Register folder created successfully in Google Drive.")

    # Fetch the model and label encoder files
    # Get the 'Models' folder ID or create it if not found
    models_folder_id = None
    results = service.files().list(q=f"'{attendance_folder_id}' in parents", fields='files(id, name)').execute()
    for item in results.get('files', []):
        if item['name'] == 'Models':
            models_folder_id = item['id']

    if not models_folder_id:
        st.error("Models folder not found in Google Drive.")
        return

    
    model_files = fetch_files_from_folder(service, models_folder_id)
    model_file = next((file for file in model_files if file['name'] == 'detect.joblib'), None)
    labels_file = next((file for file in model_files if file['name'] == 'vals.joblib'), None)

    if not model_file or not labels_file:
        st.error("Model or label encoder file not found in Google Drive.")
        return

    # Download the model and label encoder files
    model_path = 'Models/detect.joblib'
    labels_path = 'Models/vals.joblib'
    os.makedirs('Models', exist_ok=True)
    download_file(service, model_file['id'], model_path)
    download_file(service, labels_file['id'], labels_path)

    # Load the face detection model
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the trained model and label encoder
    model = joblib.load(model_path)
    vals = joblib.load(labels_path)

    # Initialize video capture
    video = cv2.VideoCapture(0)
    attendance_list = []

    st.write("Face Recognition for Attendance")
    st.write("Press 'm' to mark attendance, press 'q' to quit")

    attendance_count = 0  # Counter for number of times attendance is marked

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (128, 128))
            arr_img = np.array(resized_img)
            reshaped_img = arr_img.reshape(-1, 128, 128, 3)
            output = np.argmax(model.predict([reshaped_img]))
            if output < len(vals.classes_):
                name = vals.inverse_transform([output])[0]
                ts = time.time()
                date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                cv2.putText(frame, str(name), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                attendance_list.append([str(name), str(timestamp)])

        cv2.imshow("Mark Attendance", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):  # Press 'm' to mark attendance
            attendance_count += 1
            date = datetime.now().strftime("%d-%m-%Y")
            file_path = f"Attendance/Attendance_{date}.csv"

            # Check if the file already exists
            if os.path.exists(file_path):
                mode = 'a'  # Append mode
            else:
                mode = 'w'  # Write mode

            with open(file_path, mode, newline="") as csvfile:
                writer = csv.writer(csvfile)
                # Write header only if file is newly created
                if mode == 'w':
                    writer.writerow(['NAME', 'TIME'])
                writer.writerows(attendance_list)

            upload_file_to_drive(service, register_folder_id, file_path, os.path.basename(file_path))
            st.success("Attendance marked successfully and uploaded to Google Drive.")
            attendance_list = []  # Reset the attendance list after marking
        elif key == ord('q'):  # Press 'q' to quit
            break

    video.release()
    cv2.destroyAllWindows()

# Run the attendance marking function
if __name__ == "__main__":
    mark_attendance()

