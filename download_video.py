import yt_dlp
from yt_dlp.utils import DownloadError
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import cv2
import os

def search_and_download(query):
    ydl_opts = {
        'format': 'best',
        'noplaylist': True,
        'quiet': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Search for the query on YouTube
            search_results = ydl.extract_info(f"ytsearch:{query}", download=False)['entries']

            # Filter out None entries
            search_results = [entry for entry in search_results if entry is not None]
            if not search_results:
                print("No results found.")
                return

            # Get the first valid result
            video_info = search_results[0]
            video_url = video_info['webpage_url']

            print(f"Downloading: {video_info['title']}")

            # Download the video
            ydl.download([video_url])

            # Get the filename of the downloaded video
            video_filename = ydl.prepare_filename(video_info)
            print(f"Video downloaded as: {video_filename}")

            # Perform scene detection
            detect_scenes(video_filename)

    except DownloadError as e:
        print(f"An error occurred: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def detect_scenes(video_path):
    # Create a video manager
    video_manager = VideoManager([video_path])
    # Create a scene manager
    scene_manager = SceneManager()
    # Adjust the threshold to get 50-80 scenes
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    try:
        # Start the video manager
        video_manager.start()
        # Perform scene detection
        scene_manager.detect_scenes(frame_source=video_manager)
        # Obtain list of detected scenes
        scene_list = scene_manager.get_scene_list()
        print(f"Detected {len(scene_list)} scenes.")

        # Save images of detected scenes
        save_scene_images(video_path, scene_list)

    finally:
        # Release the video manager
        video_manager.release()

def save_scene_images(video_path, scene_list):
    # Create a directory to save scene images
    output_dir = "scene_images"
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        cap.release()
        return

    for i, scene in enumerate(scene_list):
        start_frame = scene[0].get_frames()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if ret:
            image_path = os.path.join(output_dir, f"scene_{i+1}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Saved scene image: {image_path}")
        else:
            print(f"Failed to read frame at scene {i+1}")
    cap.release()

if __name__ == "__main__":
    search_query = "super mario movie trailer"
    search_and_download(search_query)
