from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data="datasets/dataset.yaml",
    epochs=60,
    batch=22,
    imgsz=1024,
    device="cuda:1",  # Force use of GPU 1
    workers=8,
    optimizer="AdamW",
    amp=True,
    lr0=0.01
)

print("\n✅ Training of yolov8m.pt completed successfully on GPU 1!\n")
