from ultralytics import YOLO

model = YOLO("yolov8l.pt")

model.train(
    data="datasets/dataset.yaml",
    epochs=60,
    batch=14,
    imgsz=1024,
    device="cuda:2",  # Force use of GPU 2
    workers=8,
    optimizer="AdamW",
    amp=True,
    lr0=0.01
)

print("\n✅ Training of yolov8l.pt completed successfully on GPU 2!\n")
