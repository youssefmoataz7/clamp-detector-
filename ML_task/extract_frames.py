import cv2
import os
#extracts 30 random frames from the video to use for labeling and creating training data

video_path = "clamp_video.mp4"
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = total_frames / fps

print(f"Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")

# Extract every Nth frame to get ~30 spread across the video
num_to_extract = 30
step = total_frames // num_to_extract
saved = 0

for i in range(num_to_extract):
    frame_idx = i * step
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        filename = f"{output_dir}/frame_{saved:04d}.jpg"
        cv2.imwrite(filename, frame)
        saved += 1
        print(f"Saved frame {frame_idx} → {filename}")

cap.release()
print(f"\nDone. {saved} frames saved to '{output_dir}/'")