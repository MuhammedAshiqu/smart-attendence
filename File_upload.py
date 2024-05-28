# import cv2
# import os
# import pickle
# import numpy as np
# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from googleapiclient.http import MediaFileUpload

# # Function to get the path to the client secret file
# def get_client_secret_path():
#     # Assuming the client secret file is in the same directory as the script
#     return os.path.join(os.path.dirname(__file__), '.streamlit', 'client_secret_930969653781-dpcai4tesnfihldtjbup7oitdrpqij3g.apps.googleusercontent.com.json')

# # Authenticate Google Drive API
# def authenticate_google_drive():
#     SCOPES = ['https://www.googleapis.com/auth/drive']
#     creds = None
#     token_path = 'token.pickle'
#     if os.path.exists(token_path):
#         with open(token_path, 'rb') as token:
#             creds = pickle.load(token)
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(get_client_secret_path(), SCOPES)
#             creds = flow.run_local_server(port=0)
#         with open(token_path, 'wb') as token:
#             pickle.dump(creds, token)
#     return creds

# # Function to create a folder in Google Drive
# def create_folder(service, parent_id, folder_name):
#     file_metadata = {
#         'name': folder_name,
#         'mimeType': 'application/vnd.google-apps.folder',
#         'parents': [parent_id]
#     }
#     folder = service.files().create(body=file_metadata, fields='id').execute()
#     return folder.get('id')

# # Function to upload a file to Google Drive
# def upload_file(service, folder_id, file_path, file_name):
#     file_metadata = {
#         'name': file_name,
#         'parents': [folder_id]
#     }
#     media = MediaFileUpload(file_path, resumable=True)
#     file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
#     return file.get('id')

# # Function to capture face data
# def capture_data(name):
#     # Load the face detection model
#     facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     # Initialize video capture
#     video = cv2.VideoCapture(0)
#     faces_data = []
#     names_data = []
#     i = 0

#     print("Collecting data... Press 'q' to stop")

#     while True:
#         ret, frame = video.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = facedetect.detectMultiScale(gray, 1.3, 5)
#         for (x, y, w, h) in faces:
#             crop_img = frame[y:y+h, x:x+w, :]
#             resized_img = cv2.resize(crop_img, (128, 128))
#             resized_img = np.array(resized_img)
#             if len(faces_data) <= 500 and i % 10 == 0:
#                 faces_data.append(resized_img)
#                 names_data.append(name)
#             i += 1
#             cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 3)
        
#         cv2.imshow("Data Collection", frame)

#         if len(faces_data) == 500:
#             break

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     video.release()
#     cv2.destroyAllWindows()

#     faces_data = np.asarray(faces_data)
#     names_data = np.asarray(names_data)

#     # Authenticate Google Drive
#     creds = authenticate_google_drive()
#     service = build('drive', 'v3', credentials=creds)

#     folder_name = "Attendance"
#     folder_id = None

#     # Check if the Attendance folder exists, if not, create it
#     results = service.files().list(
#         q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
#         fields='files(id)').execute()
#     items = results.get('files', [])
#     if not items:
#         # Attendance folder doesn't exist, create it
#         folder_id = create_folder(service, 'root', folder_name)
#     else:
#         # Attendance folder already exists
#         folder_id = items[0]['id']

#     # Create 'data' directory if it doesn't exist
#     data_dir = 'data'
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)

#     # Save the faces data in the 'data' directory
#     faces_file_path = os.path.join(data_dir, f"faces_data.pkl")
#     with open(faces_file_path, 'wb') as file:
#         pickle.dump(faces_data, file)

#     # Save the names data in the 'data' directory
#     names_file_path = os.path.join(data_dir, f"names_data.pkl")
#     with open(names_file_path, 'wb') as file:
#         pickle.dump(names_data, file)

#     # Upload the files to the Attendance folder
#     if folder_id:
#         upload_file(service, folder_id, faces_file_path, os.path.basename(faces_file_path))
#         upload_file(service, folder_id, names_file_path, os.path.basename(names_file_path))

#     print("Data collection completed.")
    

# # Main function
# if __name__ == "__main__":
#     name = input("Enter Your Name: ")
#     try:
#         capture_data(name)
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

import cv2
import os
import pickle
import numpy as np
import streamlit as st
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload

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

# Function to create a folder in Google Drive
def create_folder(service, parent_id, folder_name):
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id]
    }
    folder = service.files().create(body=file_metadata, fields='id').execute()
    return folder.get('id')

# Function to upload a file to Google Drive
def upload_file(service, folder_id, file_path, file_name):
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

# Function to get the folder ID by name
def get_folder_id(service, parent_id, folder_name):
    query = f"'{parent_id}' in parents and name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    results = service.files().list(q=query, fields='files(id)').execute()
    items = results.get('files', [])
    if not items:
        return create_folder(service, parent_id, folder_name)
    else:
        return items[0]['id']

# Function to capture face data
def capture_data(name):
    # Load the face detection model
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize video capture
    video = cv2.VideoCapture(0)
    faces_data = []
    names_data = []
    i = 0

    st.write("Collecting data... Press 'q' to stop")

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (128, 128))
            resized_img = np.array(resized_img)
            if len(faces_data) <= 500 and i % 10 == 0:
                faces_data.append(resized_img)
                names_data.append(name)
            i += 1
            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 3)
        
        cv2.imshow("Data Collection", frame)

        if len(faces_data) == 500:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    faces_data = np.asarray(faces_data)
    names_data = np.asarray(names_data)

    # Save the captured data locally
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    faces_file_path = os.path.join(data_dir, f"{name}_faces_data.pkl")
    names_file_path = os.path.join(data_dir, f"{name}_names_data.pkl")

    with open(faces_file_path, 'wb') as faces_file:
        pickle.dump(faces_data, faces_file)

    with open(names_file_path, 'wb') as names_file:
        pickle.dump(names_data, names_file)

    # Authenticate Google Drive
    creds = authenticate_google_drive()
    service = build('drive', 'v3', credentials=creds)

    # Main Attendance folder
    folder_name = "Attendance"
    results = service.files().list(
        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
        fields='files(id)').execute()
    items = results.get('files', [])
    if not items:
        attendance_folder_id = create_folder(service, 'root', folder_name)
    else:
        attendance_folder_id = items[0]['id']

    # Subfolder for face data
    face_data_folder_id = get_folder_id(service, attendance_folder_id, 'Face_data')

    # Subfolder for name data
    name_data_folder_id = get_folder_id(service, attendance_folder_id, 'Name_data')

    # Upload the files to their respective subfolders
    upload_file(service, face_data_folder_id, faces_file_path, os.path.basename(faces_file_path))
    upload_file(service, name_data_folder_id, names_file_path, os.path.basename(names_file_path))

    st.write("Data Upload completed.")

# Main function
if __name__ == "__main__":
    st.title("Face Data Collection")

    name = st.text_input("Enter Your Name:")
    if st.button("Start Data Collection"):
        try:
            capture_data(name)
            st.write("Data collection complete.")
        except Exception as e:
            st.write(f"An error occurred: {str(e)}")
