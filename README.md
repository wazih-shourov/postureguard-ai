# PostureGuard AI

Welcome to **PostureGuard AI**, an advanced, real-time posture correction system powered by AI and computer vision.

## About the Project
This project uses your webcam to monitor your posture securely and locally. It detects body landmarks, calculates skeletal angles, and warns you via a sci-fi HUD (Heads-Up Display) overlay and audio alarms if you slouch or sit incorrectly for too long.

**Development Insights:**
- **80% Vibe Coded (AI Assisted)**: The majority of the codebase was generated using **Claude Sonet 4.6**, focusing on structure, GUI design, and boilerplate code.
- **20% Hand-Crafted Logic**: The core mathematical calculations, logic creation, threshold fine-tuning, and the connection between the AI inference and the runtime environment were manually written to ensure precision and reliability.

**Author**: Created by **Wazih Shourov** under **NextMind LAB**.

---

## Tech Stack & Machine Learning Approach
- **Python**: Core programming language.
- **MediaPipe Tasks API**: Used for state-of-the-art machine learning pose detection (Pose Landmarker). It identifies 33 3D landmarks on the human body in real-time.
- **OpenCV**: Used for webcam capturing, image processing, and rendering the futuristic Sci-Fi HUD.
- **Pygame**: Handles non-blocking audio playback for posture alarms.
- **NumPy**: Used for complex vector and angle calculations.

### Intelligent Posture Calibration Logic
Instead of relying on hardcoded "perfect" angles (which fail because everyone's setup and body are different), the system includes an adaptive **Calibration System**:
- **Learning Phase**: When you first launch the app, sit up straight. During the first 40 frames, the system analyzes your natural, upright posture to establish a personal "baseline" (calculating the median of your raw angles).
- **Delta-Based Scoring**: Once calibrated, the logic doesn't score you based on raw angles. Instead, it measures the *deviation (delta)* from your baseline. This robust approach ensures accurate posture classification whether your camera is directly in front of you or placed at an angle.

---

## Getting Started

Follow these steps to download the project from GitHub and run it on your local machine.

### 1. Clone the Repository
Download the code from GitHub to your local machine:
```bash
git clone https://github.com/YourUsername/posture-corrector.git
cd posture-corrector
```
*(Note: Replace the URL above with your actual GitHub repository URL).*

### 2. Set Up a Virtual Environment (Recommended)
It is highly recommended to create a virtual environment to keep your project dependencies completely isolated.
```bash
python -m venv venv
```
Activate the virtual environment:
- **Windows**: `venv\Scripts\activate`
- **Mac/Linux**: `source venv/bin/activate`

### 3. Install Dependencies
Install all required packages listed in the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Prepare Assets
Ensure you have the `assets/` folder in your project directory containing the expected files (like `pose_landmarker_full.task` and `beep warning sound.mp3`).

### 5. Run the Application
Start PostureGuard AI by running the main Python file:
```bash
python main.py
```

### 6. Usage Guide
1. Launch the app. A webcam window with a Jarvis-style HUD will open.
2. **Sit consistently straight for the first few seconds.** The system needs 40 frames to **calibrate** to your specific body and camera position. It will show a progress bar.
3. Once calibration reaches 100%, PostureGuard AI will actively track your neck deviation, back angle, shoulder imbalance, forward lean, and head drop.
4. If your posture deteriorates (e.g., slouching or leaning heavily into the screen) and stays "BAD" for a 5-second grace period, you will hear an audio alarm, and the screen will flash red.
5. Sitting straight again will instantly turn the status to "GOOD" and reset the alarm timer.
6. Press the **Q** key on your keyboard to securely exit the application.
