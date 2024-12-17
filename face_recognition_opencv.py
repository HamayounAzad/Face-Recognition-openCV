import cv2
import numpy as np
import os
import customtkinter as ctk
from PIL import Image, ImageTk
from datetime import datetime

class FaceRecognitionApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("OpenCV Face Recognition System")
        
        # Set fullscreen
        self.root.attributes('-fullscreen', True)
        
        # Initialize OpenCV face detection and recognition
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')  # Add eye detector
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Initialize variables
        self.cap = None
        self.is_capturing = False
        self.training_data = []
        self.labels = []
        self.label_ids = {}
        self.next_label_id = 0
        
        # Get screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate video size based on screen size (maintaining 4:3 aspect ratio)
        video_height = int(screen_height * 0.6)  # Use 60% of screen height
        video_width = int(video_height * 4/3)    # Maintain 4:3 aspect ratio
        self.video_size = (video_width, video_height)
        
        # Load existing training data if available
        self.load_training_data()
        
        # Create GUI elements
        self.setup_gui()
        
        # Bind Escape key to exit fullscreen
        self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))

    def setup_gui(self):
        # Window control buttons frame at the top
        self.control_frame = ctk.CTkFrame(self.root)
        self.control_frame.pack(side="top", fill="x", padx=5, pady=5)

        # Exit button (red)
        self.exit_button = ctk.CTkButton(
            self.control_frame,
            text="✕",
            width=40,
            height=30,
            fg_color="red",
            hover_color="darkred",
            command=self.root.quit
        )
        self.exit_button.pack(side="right", padx=5)

        # Maximize button (blue)
        self.maximize_button = ctk.CTkButton(
            self.control_frame,
            text="□",
            width=40,
            height=30,
            fg_color="blue",
            hover_color="darkblue",
            command=self.toggle_maximize
        )
        self.maximize_button.pack(side="right", padx=5)

        # Minimize button (gray)
        self.minimize_button = ctk.CTkButton(
            self.control_frame,
            text="_",
            width=40,
            height=30,
            fg_color="gray",
            hover_color="darkgray",
            command=self.root.iconify
        )
        self.minimize_button.pack(side="right", padx=5)

        # Main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Video frame
        self.video_frame = ctk.CTkFrame(self.main_frame)
        self.video_frame.pack(pady=10)

        # Create a fixed-size label for video
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.configure(width=self.video_size[0], height=self.video_size[1])
        self.video_label.pack()

        # Buttons frame
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.pack(pady=10)

        self.start_button = ctk.CTkButton(
            self.button_frame, 
            text="Start Camera", 
            command=self.toggle_camera
        )
        self.start_button.pack(side="left", padx=5)

        self.capture_button = ctk.CTkButton(
            self.button_frame, 
            text="Capture Face", 
            command=self.capture_face
        )
        self.capture_button.pack(side="left", padx=5)

        self.name_entry = ctk.CTkEntry(
            self.button_frame, 
            placeholder_text="Enter name for captured face"
        )
        self.name_entry.pack(side="left", padx=5)

        self.train_button = ctk.CTkButton(
            self.button_frame, 
            text="Start Recognition", 
            command=self.train_recognizer
        )
        self.train_button.pack(side="left", padx=5)

        # Developer credit at the bottom
        self.credit_label = ctk.CTkLabel(
            self.main_frame,
            text="Developed By: Mohammad Hamayoun Azad",
            font=("Arial", 12)
        )
        self.credit_label.pack(side="bottom", pady=10)

    def load_training_data(self):
        if not os.path.exists("dataset"):
            os.makedirs("dataset")
        
        if os.path.exists("trainer.yml"):
            self.face_recognizer.read("trainer.yml")
            
        # Load label mappings
        if os.path.exists("labels.txt"):
            with open("labels.txt", "r") as f:
                for line in f:
                    name, label_id = line.strip().split(":")
                    self.label_ids[name] = int(label_id)
                    self.next_label_id = max(self.next_label_id, int(label_id) + 1)

    def save_label_mappings(self):
        with open("labels.txt", "w") as f:
            for name, label_id in self.label_ids.items():
                f.write(f"{name}:{label_id}\n")

    def toggle_camera(self):
        if not self.is_capturing:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Added CAP_DSHOW for Windows
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                return
            
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size[1])
            
            self.is_capturing = True
            self.start_button.configure(text="Stop Camera")
            self.update_video()
        else:
            self.is_capturing = False
            self.start_button.configure(text="Start Camera")
            if self.cap is not None:
                self.cap.release()
            self.video_label.configure(image=None)

    def update_video(self):
        if self.is_capturing and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Resize frame to match our video size
                frame = cv2.resize(frame, self.video_size)
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces with stricter parameters
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=8,  # Increased for stricter detection
                    minSize=(60, 60),  # Increased minimum face size
                    maxSize=(400, 400)  # Added maximum face size
                )

                # Process each face
                for (x, y, w, h) in faces:
                    face_roi_gray = gray[y:y+h, x:x+w]
                    
                    # Detect eyes in the face region
                    eyes = self.eye_detector.detectMultiScale(
                        face_roi_gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(20, 20)
                    )
                    
                    # Only process faces that have eyes (confirms it's a human face)
                    if len(eyes) >= 1:
                        # Draw rectangle around face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Try to recognize face
                        try:
                            face_roi = gray[y:y+h, x:x+w]
                            label_id, confidence = self.face_recognizer.predict(face_roi)
                            
                            # Find name associated with label_id
                            name = "Unknown"
                            for person_name, pid in self.label_ids.items():
                                if pid == label_id and confidence < 70:  # Lower confidence means better match
                                    name = person_name
                                    break
                            
                            # Display name and confidence
                            confidence_text = f"{100 - confidence:.1f}%" if confidence < 100 else "Unknown"
                            cv2.putText(frame, f"{name} ({confidence_text})", 
                                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                        except:
                            pass

                # Convert frame for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image=image)
                
                self.video_label.configure(image=photo)
                self.video_label.image = photo  # Keep a reference!

            if self.is_capturing:  # Check again in case camera was stopped
                self.root.after(10, self.update_video)

    def capture_face(self):
        if not self.is_capturing:
            return

        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(60, 60),
                maxSize=(400, 400)
            )

            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                (x, y, w, h) = largest_face
                
                face_roi_gray = gray[y:y+h, x:x+w]
                
                # Verify it's a human face by detecting eyes
                eyes = self.eye_detector.detectMultiScale(
                    face_roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20)
                )
                
                if len(eyes) >= 1:
                    name = self.name_entry.get()
                    if not name:
                        return

                    # Ensure directory exists
                    if not os.path.exists("dataset"):
                        os.makedirs("dataset")

                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"dataset/{name}_{timestamp}.jpg"
                    cv2.imwrite(filename, face_roi_gray)

                    # Update known faces
                    if name not in self.label_ids:
                        self.label_ids[name] = self.next_label_id
                        self.next_label_id += 1
                    
                    self.save_label_mappings()
                    self.name_entry.delete(0, 'end')

    def train_recognizer(self):
        faces = []
        labels = []
        
        # Load all face images
        for name in os.listdir("dataset"):
            person_dir = os.path.join("dataset", name)
            if os.path.isdir(person_dir):
                label_id = self.label_ids.get(name)
                if label_id is not None:
                    for image_name in os.listdir(person_dir):
                        image_path = os.path.join(person_dir, image_name)
                        face_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        faces.append(face_img)
                        labels.append(label_id)
        
        if faces and labels:
            # Train the recognizer
            self.face_recognizer.train(faces, np.array(labels))
            self.face_recognizer.save("trainer.yml")

    def toggle_maximize(self):
        is_fullscreen = self.root.attributes('-fullscreen')
        if is_fullscreen:
            self.root.attributes('-fullscreen', False)
            self.maximize_button.configure(text="□")
        else:
            self.root.attributes('-fullscreen', True)
            self.maximize_button.configure(text="❐")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run()
