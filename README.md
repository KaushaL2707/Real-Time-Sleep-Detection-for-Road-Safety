# Real-Time-Sleep-Detection-for-Road-Safety


## Overview
This project is a Python-based application that uses a webcam to monitor a driver's eye state. The system detects if the driver's eyes remain closed for a prolonged duration and plays an alert sound to prevent accidents caused by drowsiness.

## Features
- Detects the driver's eyes using a webcam and Mediapipe's Face Mesh module.
- Calculates the Eye Aspect Ratio (EAR) to determine if the eyes are open or closed.
- Triggers an alarm when the eyes remain closed for more than a threshold duration.
- Real-time visualization with a live feed, including visual indicators for eye states and a score counter.

## Dependencies
- Python 3.7+
- OpenCV
- Mediapipe
- NumPy
- Pygame

## Installation
1. Clone the repository or download the project files.
2. Install the required Python libraries:
   ```bash
   pip install opencv-python mediapipe numpy pygame
   ```
3. Place the alarm sound file (`mixkit-alert-alarm-1005.wav`) in the same directory as the script.

## How It Works
1. The system captures video input from the webcam.
2. Mediapipe's Face Mesh module detects the face landmarks, including the eyes.
3. The Eye Aspect Ratio (EAR) is calculated based on the landmarks to determine the state of the eyes (open or closed).
4. If the eyes remain closed for a predefined duration (measured by a score counter), an alert sound is triggered.
5. The live feed displays real-time annotations, such as eye bounding circles and the current score.

### Key Components
- **Eye Aspect Ratio (EAR):**
  Calculates the vertical and horizontal distances of the eye landmarks to detect eye closure.
- **Score Counter:**
  Increments when eyes are closed and decrements when open. If the score exceeds 300, the alarm is triggered.
- **Visual Indicators:**
  Displays eye state (“Eyes Closed” or “Eyes Open”), score, and bounding circles around the eyes.

### Alarm System
The alarm is implemented using the Pygame mixer module. The sound plays when the score exceeds a certain threshold.

## Usage
1. Run the script:
   ```bash
   python app.py
   ```
2. The application will open the webcam feed.
3. Observe the live feed for visual indicators of eye state and the score.
4. If the alarm triggers, it indicates the driver’s eyes have been closed for too long.
5. Press `q` to exit the application.


## Note
- Ensure the webcam is properly connected and functional.
- Adjust the `EAR_THRESHOLD` and score logic based on individual preferences or environmental conditions.
- The project requires a suitable environment with proper lighting for accurate detection.

## Limitations
- May not work accurately in poor lighting conditions.
- Glasses or other obstructions may affect eye detection.
- Not optimized for multi-face detection.

## Future Improvements
- Add support for night-time or infrared cameras.
- Enhance detection for drivers wearing glasses or sunglasses.
- Optimize the system for better performance on low-power devices.

## License
This project is open-source and available for educational and non-commercial purposes.

