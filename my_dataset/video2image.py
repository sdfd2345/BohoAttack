import cv2
import os
def video_to_images(video_path, output_folder):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0

    while True:
        # Read the next frame
        ret, frame = video.read()

        # If no frame is returned, the video has ended
        if not ret:
            break

        # Save the frame as an image
        output_path = os.path.join(output_folder, f"{frame_count:05d}.jpg")
        cv2.imwrite(output_path, frame)

        frame_count += 1

    # Release the video file
    video.release()

# Example usage
video_path = "/home/yjli/AIGC/Adversarial_camou/my_dataset/A1.mp4"
output_folder = "/home/yjli/AIGC/Adversarial_camou/my_dataset/A1"

video_to_images(video_path, output_folder)