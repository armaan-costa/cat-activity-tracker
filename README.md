# 🐱 Cat Activity Tracker

A **real-time cat activity classifier** using PyTorch, OpenCV, and your webcam.  
It predicts your cat's behavior (`sleeping`, `eating`, `resting`, `walking`, `sitting`, `grooming`, and `catloaf`) live, displaying the predicted label on a clean, resizable camera feed.

Feel free to try and test this on your own cats!

I built this project to learn PyTorch and practice building a model training script, while also running a fun experiment collecting and labeling data of my own two cats for a supervised learning model.

This project is open to contributions if you want to try and make the tracker do other cool stuff like also detect stretching or playing, detect the cat in the image, or keep track of cat activities in an output log.

---

## 🎥 Demo
[Watch on YouTube](https://youtu.be/aLzh6RstDZg?si=TQOziImodNNM5cj8) | [Watch on Twitter/X](https://x.com/armaancosta/status/1943174411001151987)

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
