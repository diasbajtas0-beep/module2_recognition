import cv2
from pathlib import Path


def save_video(frames_dir):
    #frames_dir = Path(path)
    output_video = "output4.mp4"

    # sorted list of frames
    frames = sorted(frames_dir.glob("frame_*.jpg"))

    # read first frame to get size
    first = cv2.imread(str(frames[0]))
    h, w, _ = first.shape

    fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        writer.write(img)

    writer.release()


    print(f"{output_video} created")