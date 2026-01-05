# 🤖 Anti-Spoofing Face Attendance System (6000+ Users)

A production-ready Face Recognition & Attendance System designed for scale. It integrates **DeepFace (ArcFace)** for recognition and **Gaze-Based Liveness Detection** for anti-spoofing, optimized for both CPU and GPU environments.

---

## 🚀 Key Features

*   **⚡ High Performance**: Multi-threaded architecture enables real-time recognition for **6,000+ identities**.
*   **🛡️ Anti-Spoofing**: Interactive **Eye-Gaze Liveness Detection** prevents phone/photo spoofing attacks.
*   **🧠 Scalable Database**: Uses **FAISS** (Facebook AI Similarity Search) for millisecond-speed vector lookups.
*   **🖥️ Hardware Agnostic**: Auto-switches between **CPU** (for compatibility) and **GPU** (for speed) based on available hardware.
*   **🛠️ CLI Tools**: Includes robust command-line tools for training, adding users, and calibration.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/HarshitWorkGit/Anti-Spoofing-Attendance-System.git
cd Anti-Spoofing-Attendance-System
```

### 2. Install Dependencies (Strict Mode)
It is critical to install the **exact versions** specified to ensure model and TensorFlow compatibility.
```bash
# Force install exact versions from requirements.txt
pip install -r requirements.txt --force-reinstall --no-deps
```
*Note: Using a python virtual environment (Python 3.10/3.11) is highly recommended.*

---

## 📂 Usage

### 1️⃣ Run Attendance System
Launch the main attendance processing engine.
```bash
python run_attendance.py
```
*   **Locked Identity**: Recognizes face first, then locks on to verify liveness.
*   **Green/Red Output**: Visual feedback on terminal for attendance status.

### 2️⃣ Add New Users (Fast)
To add users without re-training the entire 6000-person dataset:
1.  Create a folder: `Add New Identity/John_Doe/`.
2.  Paste 3-5 clear face photos inside.
3.  Run the script:
    ```bash
    python "Add New Identity/add_new_identity.py"
    ```
4.  The system will extract vectors, update the FAISS index, and move images to the main dataset.

### 3️⃣ Train from Scratch
If you want to rebuild the model from the full `dataset/` folder:
```bash
python train_model.py
```
*   Follow the interactive prompts to select Model (ArcFace/FaceNet) and Device (CPU/GPU).

### 4️⃣ Camera Calibration
Eye-gaze detection works best when calibrated to your specific camera and screen setup.
```bash
python camera_calibration/calibrate.py
```
*   Follow the red dot on the screen with your eyes to generate a `calibration.json` file.

---

## 📂 Directory Structure

```text
├── dataset/                  # Main database of user images
├── models/                   # Stores trained FAISS indices (.bin) and labels (.npy)
├── camera_calibration/       # Calibration tools and configuration
├── Add New Identity/         # Drop folder for new user registration
├── run_attendance.py         # Main Execution Script
└── train_model.py            # Training Script
```
