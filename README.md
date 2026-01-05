# Face Recognition & Anti-Spoofing Attendance System

A high-performance Face Attendance System capable of scaling to **6000+ identities** using **ArcFace**, **FAISS**, and **Silent-Face-Anti-Spoofing**.

## 🚀 Features
- **Scalable**: Verified with 6000+ identities.
- **Fast**: Multi-threaded recognition pipeline.
- **Anti-Spoofing**: Eye-tracking liveness detection to prevent photo attacks.
- **Interactive Training**: CLI tools to easily train models and add new users.
- **CPU/GPU Optimized**: Automatically selects the best hardware.

## 📂 Project Structure
- `run_attendance.py`: Main script to run the attendance system.
- `train_model.py`: Train ArcFace/FaceNet models from scratch.
- `add_new_identity.py`: Add new users without retraining the whole dataset.
- `camera_calibration/`: Tools to calibrate eye-tracking.

## 🛠️ Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place images in `dataset/` (structure: `dataset/person_name/image.jpg`).
3. Train the model:
   ```bash
   python train_model.py
   ```
4. Run attendance:
   ```bash
   python run_attendance.py
   ```

## 📸 Usage
- **Registration**: Put new user folders in `Add New Identity/` and run `python add_new_identity.py`.
- **Calibration**: Run `python camera_calibration/calibrate.py` once to tune eye tracking for your camera.
