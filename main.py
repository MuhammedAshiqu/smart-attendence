import streamlit as st
from attendance import mark_attendance
from File_upload import capture_data
from train_model import train_model

# Define main function
def main():
    st.title("Face Recognition Attendance System")

    menu = ["Home", "Mark Attendance", "Upload Data", "Train Model"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.write("Welcome to the Face Recognition Attendance System!")
    elif choice == "Mark Attendance":
        st.subheader("Mark Attendance")
        if st.button("Start Marking Attendance"):
            mark_attendance()
    elif choice == "Upload Data":
        st.subheader("Upload Data")
        name = st.text_input("Enter Your Name:")
        if st.button("Start Data Collection"):
            if name:
                capture_data(name)
            else:
                st.warning("Please enter a name.")
    elif choice == "Train Model":
        st.subheader("Train Model")
        if st.button("Start Training"):
            train_model()

# Run main function
if __name__ == "__main__":
    main()
