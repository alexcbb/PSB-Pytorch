import os 
import cv2
import argparse


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")

    args = parser.parse_args()

    # Load the files from the data directory
    files = os.listdir(args.data_dir)
    print(f"Processing {len(files)} files")
    for file in files:
        print(f"Extracting frames from {file}")
        filename = file.split(".")[0]
        path = os.path.join(args.data_dir, file)
        video = cv2.VideoCapture(path)
        # Check if the video is opened
        if not video.isOpened():
            print("Error: Could not open video.")
            exit()
        # Create a directory to store the frames
        os.makedirs(f"data/frames_{filename}", exist_ok=True)
        # Read the video frame by frame
        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            # Save the frame
            cv2.imwrite(f"data/frames_{filename}/frame_{frame_count:04d}.jpg", frame)
            frame_count += 1
        print(f"Total frames: {frame_count}")
        # Release the video
        video.release()