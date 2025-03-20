from ultralytics import YOLO

model = YOLO("yolov8x.pt")

model.train(
    data="datasets/dataset.yaml",
    epochs=60,
    batch=10,
    imgsz=1024,
    device="cuda:3",  # Force use of GPU 3
    workers=8,
    optimizer="AdamW",
    amp=True,
    lr0=0.01
)

print("\n✅ Training of yolov8x.pt completed successfully on GPU 3!\n")
