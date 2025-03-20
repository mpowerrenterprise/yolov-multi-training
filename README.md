# 🚀 YOLOv Multi-GPU Training

An efficient multi-GPU training setup for YOLOv models, utilizing PyTorch DistributedDataParallel (DDP) to scale training across multiple GPUs.

## 🛠 Features
- Multi-GPU Training – Train YOLOv models efficiently on multiple GPUs.
- Scalability – Supports single-node multi-GPU and multi-node distributed training.
- Optimized Performance – Load balancing, gradient synchronization, and mixed-precision training.


When training multiple YOLO models on different GPUs in AWS or remote servers, using tmux ensures that the training continues even if the SSH connection is lost. This guide explains how to set up, manage, and monitor multi-GPU training using tmux.

## Why Use tmux?

- Prevents training interruptions when SSH disconnects.

- Allows multiple processes (each model on a different GPU).

- Easy to reattach and monitor training progress.


## Setup


### 01: Install Ultralytics

```
pip install ultralytics
```

### 02: Create a Tmux session

```
tmux new -s training
```

### 03: Run scripts

```
nohup python train_0.py > train_0.log 2>&1 &
nohup python train_1.py > train_1.log 2>&1 &
nohup python train_2.py > train_2.log 2>&1 &
nohup python train_3.py > train_3.log 2>&1 &
```

### 04: Detach from tmux

```
CTRL + B, then press D
```

### 05: Reattach to the session

```
tmux attach -t training
```

# Contact

## 🌐 Website:
[![Visit](https://img.shields.io/badge/Visit%3A%20www.mpowerr.com-%23007ACC?style=flat&logo=google-chrome&logoColor=white&labelWidth=200)](https://www.mpowerr.com)

---

## 📱 Social Media:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/mpowerr-info)
[![Facebook](https://img.shields.io/badge/Facebook-%231877F2?style=for-the-badge&logo=facebook&logoColor=white)](https://www.facebook.com/mpowerr.info)
[![Instagram](https://img.shields.io/badge/Instagram-%23E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/mpowerr.info)
[![X (Twitter)](https://img.shields.io/badge/X-%231DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/MpowerrInfo)
[![TikTok](https://img.shields.io/badge/TikTok-%23000000?style=for-the-badge&logo=tiktok&logoColor=white)](https://www.tiktok.com/@mpowerr.info)
[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@mpowerrinfo)

---