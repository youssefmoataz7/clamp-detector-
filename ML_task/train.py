from ultralytics import YOLO
#for local training

model = YOLO("yolo11m.pt")

results = model.train(
    data="/Users/youssefmoataz/Development/python projects/ML_task/clamp-detection/data.yaml",  # path from Roboflow export
    epochs=50,
    imgsz=640,
    batch=8,
    name="clamp_detector",
    patience=10,
    save=True
)

print("Training complete!")
print(f"Best weights saved to: {results.save_dir}/weights/best.pt")

