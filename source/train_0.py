from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="datasets/dataset.yaml",
    epochs=60,
    batch=38,
    imgsz=1024,
    device="cuda:0",  # Force use of GPU 0
    workers=8,
    optimizer="AdamW",
    amp=True,
    lr0=0.01
)

print("\n✅ Training of yolov8s.pt completed successfully on GPU 0!\n")
