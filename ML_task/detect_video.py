from ultralytics import YOLO
#tests the models accuracy on the video

model = YOLO("runs/detect/clamp_detector/weights/best.pt")

# Run inference directly on video — YOLO handles frame-by-frame
results = model.predict(
    source="clamp_video.mp4",
    conf=0.4,           # confidence threshold (tweak if needed)
    iou=0.5,            # NMS overlap threshold
    save=True,          # saves output video automatically
    name="clamp_output"
)

print("Detection done!")
print("Output video saved in: runs/detect/clamp_output/")