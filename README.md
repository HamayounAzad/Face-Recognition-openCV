# AI Face Recognition System

A sophisticated real-time face recognition system built with Python, OpenCV, and CustomTkinter. This application provides a modern GUI interface for face detection, recognition, and management with high accuracy and real-time performance.

## 🌟 Features

- **Real-time Face Detection & Recognition**
  - Advanced human face detection using Haar Cascade Classifiers
  - Eye detection verification to ensure only human faces are processed
  - Real-time confidence score display
  - High-accuracy LBPH (Local Binary Pattern Histogram) Face Recognition

- **Modern User Interface**
  - Full-screen capable interface
  - Window control buttons (Minimize, Maximize, Exit)
  - Live video feed display
  - Intuitive controls for face capture and recognition

- **Smart Face Processing**
  - Automatic face data management
  - Multiple face detection and recognition
  - Confidence score display for recognition accuracy
  - Organized dataset storage system

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-face-recognition.git
   cd ai-face-recognition
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Requirements

- Python 3.8 or higher
- OpenCV
- NumPy
- CustomTkinter
- Pillow

## 💻 Usage

1. Run the application:
   ```bash
   python face_recognition_opencv.py
   ```

2. **Controls:**
   - Press `ESC` to exit fullscreen mode
   - Use window control buttons for minimize, maximize, and close
   - Enter name and click "Capture Face" to add new faces
   - Click "Start Recognition" to begin face recognition

3. **Adding New Faces:**
   - Enter the person's name in the text field
   - Ensure good lighting and face positioning
   - Click "Capture Face" to save
   - Multiple angles recommended for better recognition

## 🎯 Key Features Explained

### Face Detection
- Uses Haar Cascade Classifier for initial face detection
- Implements eye detection for verification
- Strict parameter tuning for accurate human face detection
- Size constraints to filter out invalid detections

### Face Recognition
- LBPH Face Recognizer for robust recognition
- Confidence score display for transparency
- Organized data storage in dataset directory
- Automatic label management system

### User Interface
- Full-screen capable display
- Modern, intuitive controls
- Real-time video feed
- Window management buttons
- Status indicators and confidence displays

## 📁 Project Structure

```
ai-face-recognition/
│
├── face_recognition_opencv.py    # Main application file
├── requirements.txt             # Project dependencies
├── README.md                   # Project documentation
│
├── dataset/                    # Stored face data
│   └── [person_name]/         # Individual face images
│
└── trainer.yml                # Trained recognition model
```

## ⚙️ Configuration

The system includes several configurable parameters:

- Face Detection Parameters:
  - `scaleFactor`: 1.1
  - `minNeighbors`: 8
  - `minSize`: (60, 60)
  - `maxSize`: (400, 400)

- Recognition Parameters:
  - Confidence threshold: 70%
  - Eye detection verification
  - Multiple face processing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📌 Version Control

### Version 1.0.0 (Current)
- Initial release
- Core features implemented:
  - Real-time face detection with human verification
  - LBPH Face Recognition system
  - Modern GUI with full-screen support
  - Window control buttons
  - Face capture and training system
  - Dataset management

### Future Updates (Planned)
#### Version 1.1.0
- [ ] Multiple face recognition optimization
- [ ] Enhanced GUI themes
- [ ] Performance improvements
- [ ] Additional face angles support

#### Version 1.2.0
- [ ] Batch face registration
- [ ] Export/Import dataset functionality
- [ ] Recognition history logging
- [ ] Advanced settings configuration

### Changelog
#### 1.0.0 (December 2024)
- Released initial version
- Implemented core face recognition functionality
- Added GUI with modern design
- Integrated eye detection for human verification
- Added dataset management system
- Implemented real-time confidence score display
- Added window control functionality
- Created comprehensive documentation

### Branch Structure
- `main`: Stable release version
- `develop`: Development and testing
- `feature/*`: New feature development
- `bugfix/*`: Bug fixes
- `release/*`: Release preparation

### Commit Convention
We follow the conventional commits specification:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation updates
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test updates
- `chore`: Maintenance tasks

### Contributing Guidelines
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'feat: Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Important License Notice

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). **Before using this software, you MUST read and comply with the LICENSE file**. Key requirements include:
- Mandatory attribution to Mohammad Hamayoun Azad
- Inclusion of original author's contact information
- Clear indication of any modifications
- Proper attribution in all distributed materials

See the LICENSE file for complete terms and requirements.

## 📝 License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) - see the LICENSE file for details.

## 👨‍💻 Developer

Developed by: Mohammad Hamayoun Azad

## 📧 Contact

For any queries or suggestions, please reach out to:
[whatsapp](https://wa.me/93700230047)

---
⭐ Star this repository if you find it helpful!
