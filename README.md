# 🚦 Traffic Violation Detection System

An AI-based Traffic Monitoring System that detects vehicles, tracks their movement, identifies traffic violations (overspeeding & wrong lane), and extracts number plate information using Computer Vision.

---

## 📌 Project Overview

This project processes recorded road traffic videos and performs:

- Vehicle Detection & Classification
- Multi-Object Tracking
- Lane Detection
- Speed Estimation (Relative Speed)
- Wrong Lane Violation Detection
- Number Plate Recognition using OCR
- Annotated Output Video Generation

The system highlights violating vehicles with red bounding boxes and displays violation type along with detected plate number.

---

## 🛠️ Tech Stack

- Python
- OpenCV
- YOLO (Ultralytics)
- EasyOCR
- NumPy

---

## 🧠 How It Works

1. Input recorded traffic video
2. Extract frames using OpenCV
3. Detect and track vehicles using YOLO
4. Detect lane boundaries using edge detection and Hough Transform
5. Estimate vehicle speed using pixel displacement across frames
6. Detect violations based on predefined rules
7. Extract number plate text using EasyOCR
8. Generate annotated output video

---

## 🚗 Detected Violations

- Overspeeding (Relative Speed Based)
- Wrong Lane Movement

---

## 📊 Output

- Vehicle Type
- Lane Number
- Speed Value
- Violation Type
- Number Plate (for violating vehicles)
- Annotated output video with bounding boxes

---

## ⚠️ Note

- Speed values are relative (pixel-based), not real-world km/h.
- OCR accuracy depends on video quality and lighting conditions.
- The system runs on CPU (GPU acceleration optional).

---

## 🚀 Future Improvements

- Real-world speed calibration
- Red light violation detection
- Accident detection
- Database logging of violations
- Live CCTV integration

---

## 👨‍💻 Author

Haari Murthy  
B.Tech Computer Science & Engineering (AI Specialization)

---

⭐ If you found this project interesting, feel free to give it a star!
