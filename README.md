### Smart Attendence System
#### Attendence.py

This script uses face recognition technology to mark attendance automatically. It integrates with Google Drive to fetch and upload relevant files and data. Key features include:

- **Google Drive Integration**: Authenticates with Google Drive API to access and manage files.
- **Face Detection and Recognition**: Uses OpenCV and a pre-trained model to detect faces from a webcam feed and recognize them.
- **Attendance Logging**: Records attendance in a CSV file and uploads it to Google Drive.
- **Real-Time Video Capture**: Captures video feed from a webcam to detect and identify faces.
- **Streamlit Integration**: Uses Streamlit for the user interface to display messages and status updates.

##### Detailed Breakdown:

1. **Google Drive Authentication**:
   - Authenticates using OAuth 2.0 and saves the credentials for future use.
   
2. **File Management on Google Drive**:
   - Fetches model files and label encoders from a specified Google Drive folder.
   - Uploads the attendance CSV file to Google Drive after marking attendance.

3. **Face Detection and Recognition**:
   - Utilizes OpenCV's `CascadeClassifier` for face detection.
   - Loads a pre-trained model and label encoder using `joblib` to recognize faces.

4. **Attendance Marking**:
   - Captures frames from the webcam, detects faces, and recognizes them.
   - Records the recognized names and timestamps in an attendance list.
   - Saves the attendance data to a CSV file and uploads it to Google Drive.

5. **Streamlit Interface**:
   - Provides a user interface to interact with the script, including messages for errors and success.

##### Usage Instructions:

- Run the script to start the attendance marking process.
- Ensure that the required model and label encoder files are present in the Google Drive folder.
- Use the Streamlit interface to monitor progress and view messages.
- Press 'm' to mark attendance and 'q' to quit the video feed.

