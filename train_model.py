import os
import pickle
import streamlit as st
import numpy as np
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import Callback
import pandas as pd
import joblib

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
def download_file(service, file_id):
    request = service.files().get_media(fileId=file_id)
    file_data = io.BytesIO()
    downloader = MediaIoBaseDownload(file_data, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    file_data.seek(0)
    return file_data

# Function to display information about face and name data
def show_face_data_info(names, face_files, name_files, service):
    user_names = []
    face_data_lengths = []
    name_data_lengths = []
    
    # Fetch face data lengths
    for name, file in zip(names, face_files):
        file_data = download_file(service, file['id'])
        faces_data = pickle.load(file_data)
        user_names.append(name)
        face_data_lengths.append(len(faces_data))
    
    # Fetch name data lengths
    for file in name_files:
        file_data = download_file(service, file['id'])
        names = pickle.load(file_data)
        name_data_lengths.append(len(names))
    
    # Display data lengths in a table
    data = list(zip(user_names, face_data_lengths, name_data_lengths))
    df = pd.DataFrame(data, columns=["Name", "Face_data", "Name_data"])
    st.table(df)

    return face_data_lengths, name_data_lengths

# Custom callback to update progress bar
class StreamlitProgressBarCallback(Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.progress_bar = st.progress(0)
        self.current_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1
        progress = self.current_epoch / self.epochs
        self.progress_bar.progress(progress)

# Function to upload a file to Google Drive
def upload_file_to_drive(service, folder_id, file_path, file_name):
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

# Function to save model and encoder to Google Drive
def save_model_and_encoder_to_drive(service, model, label_encoder, folder_id):
    model_path = 'detect.joblib'
    labels_path = 'vals.joblib'
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, labels_path)

    model_file_id = upload_file_to_drive(service, folder_id, model_path, 'detect.joblib')
    labels_file_id = upload_file_to_drive(service, folder_id, labels_path, 'vals.joblib')
    return model_file_id, labels_file_id

# Function to train the model
def train_model(X, y, service, models_folder_id):
    if len(X) != len(y):
        st.error("X and y arrays must have the same length.")
        return
    
    input_shape = (128, 128, 3)

    # Create the model
    model = Sequential()

    # Add layers to the model
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense((len(X) // 1), activation='softmax'))
    # Using integer division for the number of classes

    # Display the model summary
    model.summary()
    model.compile(loss='SparseCategoricalCrossentropy', optimizer='adam', metrics=['accuracy'])
    label_encoder = LabelEncoder()
    # Fit and transform labels
    y_encoded = label_encoder.fit_transform(y)
    
    epochs = 10
    progress_callback = StreamlitProgressBarCallback(epochs)
    
    trained_history = model.fit(X, y_encoded, batch_size=100, epochs=epochs, callbacks=[progress_callback])
    tr_loss, tr_accuracy = model.evaluate(X, y_encoded)
    st.write('Train Accuracy: ', tr_accuracy * 100)
    st.write('Train Loss: ', tr_loss * 100)
    if tr_accuracy >= 0.50:
        st.write('Model trained successfully!')
        save_model_and_encoder_to_drive(service, model, label_encoder, models_folder_id)
    else:
        st.error('Error: Model did not meet the desired accuracy.')

# Main function
def main():
    st.title("Model Training")

    creds = authenticate_google_drive()
    service = build('drive', 'v3', credentials=creds)

    attendance_folder_name = "Attendance"
    results = service.files().list(
        q=f"name='{attendance_folder_name}' and mimeType='application/vnd.google-apps.folder'",
        fields='files(id)').execute()
    items = results.get('files', [])
    if not items:
        st.error("Attendance folder not found in Google Drive.")
        return
    attendance_folder_id = items[0]['id']

    # Get Face_data and Name_data folder IDs
    face_data_folder_id = None
    name_data_folder_id = None
    models_folder_id = None

    results = service.files().list(q=f"'{attendance_folder_id}' in parents", fields='files(id, name)').execute()
    for item in results.get('files', []):
        if item['name'] == 'Face_data':
            face_data_folder_id = item['id']
        elif item['name'] == 'Name_data':
            name_data_folder_id = item['id']
        elif item['name'] == 'Models':
            models_folder_id = item['id']

    if not face_data_folder_id or not name_data_folder_id:
        st.error("Face_data or Name_data folder not found in Google Drive.")
        return

    # Create Models folder if it doesn't exist
    if not models_folder_id:
        file_metadata = {
            'name': 'Models',
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [attendance_folder_id]
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        models_folder_id = folder.get('id')

    # Fetch all face_data.pkl files
    face_files = fetch_files_from_folder(service, face_data_folder_id)
    name_files = fetch_files_from_folder(service, name_data_folder_id)

    names = []
    for file in name_files:
        file_data = download_file(service, file['id'])
        name_data = pickle.load(file_data)
        unique_names = np.unique(name_data)
        names.extend(unique_names)

    # Display names in multiselect widget
    options = st.multiselect("Select names to include in the model training:", names)

    if st.button("Show Data"):
        # st.write(f"Selected names: {options}")
        # Show the number of items in each face_data.pkl file corresponding to the selected names
        selected_face_files = [file for file in face_files if file['name'].split('_')[0] in options]
        face_lengths, name_lengths = show_face_data_info(options, selected_face_files, name_files, service)
        
        train_option = st.radio("Do you want to train the model?", ('Yes', 'No'))

        if train_option == 'Yes':
            # Load data for training
            X = []
            y = []
            for file in selected_face_files:
                file_data = download_file(service, file['id'])
                faces_data = pickle.load(file_data)
                X.extend(faces_data)
                y.extend([file['name'].split('_')[0]] * len(faces_data))

            X = np.array(X)
            y = np.array(y)
            X = X / 255

            # Train the model
            train_model(X, y, service, models_folder_id)
        else:
            st.write("Model training skipped.")

    st.write("Thank you.")

if __name__ == "__main__":
    main()
