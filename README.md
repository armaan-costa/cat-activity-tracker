# 🐱 Cat Activity Tracker

A **real-time cat activity classifier** using PyTorch, OpenCV, and your webcam.  
It predicts your cat's behavior (`sleeping`, `eating`, `resting`, `walking`, `sitting`, `grooming`, and `catloaf`) live, displaying the predicted label on a clean, resizable camera feed.

I built this project to learn PyTorch and practice building a model training script, while also running a fun experiment collecting and labeling data of my own two cats for a supervised learning model.

---

## 🎥 Demo
[![YouTube Link](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)]([https://www.youtube.com/watch?v=YOUR_VIDEO_ID](https://www.youtube.com/watch?v=aLzh6RstDZg))

![ash-sleep](https://github.com/user-attachments/assets/8201c078-08d8-47fc-8e72-014267fe1153)
![ore-grooming](https://github.com/user-attachments/assets/20d6ae4a-a7e1-4314-9d3d-f6186d068a36)

---

## 📦 Features

✅ Uses **PyTorch** with transfer learning (ResNet18)  

✅ Real-time webcam activity tracking  

✅ Easily expandable with your own cats and behavior labels

✅ Clean overlay

---

## 🖥️ Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- Pillow (PIL)

Install dependencies:

```bash
pip install torch torchvision opencv-python pillow
