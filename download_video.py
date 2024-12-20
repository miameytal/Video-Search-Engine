import yt_dlp
from yt_dlp.utils import DownloadError
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import cv2
import os
import moondream as md
from PIL import Image
import hashlib

# Retrieve the model path from an environmental variable
model_path = os.getenv('MOONDREAM_MODEL_PATH')
model = md.vl(model=model_path)

def search_and_download(query):
    # Create a unique filename for the JSON file based on the query
    query_hash = hashlib.md5(query.encode()).hexdigest()
    json_filename = f'scene_captions_{query_hash}.json'

    if os.path.exists(json_filename):
        print(f"{json_filename} already exists; skipping download and caption generation.")
        return

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
            detect_scenes(video_filename, json_filename)

    except DownloadError as e:
        print(f"An error occurred: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def detect_scenes(video_path, json_filename):
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
        save_scene_images(video_path, scene_list, json_filename)

    finally:
        # Release the video manager
        video_manager.release()

def save_scene_images(video_path, scene_list, json_filename):
    # Create a directory to save scene images
    output_dir = "scene_images"
    os.makedirs(output_dir, exist_ok=True)

    # Determine the number of digits required to represent the scene numbers
    num_scenes = len(scene_list)
    num_digits = len(str(num_scenes))

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
            # Ensure the scene number is represented with the appropriate number of digits
            scene_number = str(i + 1).zfill(num_digits)
            image_path = os.path.join(output_dir, f"scene_{scene_number}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Saved scene image: {image_path}")
        else:
            print(f"Failed to read frame at scene {i+1}")
    cap.release()

    # Generate captions for the saved scene images
    generate_caption(output_dir, json_filename)

def generate_caption(output_dir, json_filename):
    import json
    if os.getenv('USE_CAPTION_STUB') == 'true':
        # Produce dummy captions for each image in the directory
        images = [f for f in sorted(os.listdir(output_dir)) if f.endswith('.jpg')]
        captions_dict = {}
        for i, image_file in enumerate(images, start=1):
            captions_dict[i] = f"Dummy caption {i}"
            print(f"DUMMY CAPTION for {image_file}: {captions_dict[i]}")

        with open(json_filename, 'w') as f:
            json.dump(captions_dict, f, indent=4)
        print("Dummy captions generated. JSON file created.")
        return

    captions_dict = {}
    # Iterate over each saved scene image
    for i, image_file in enumerate(sorted(os.listdir(output_dir)), start=1):
        image_path = os.path.join(output_dir, image_file)
        if os.path.isfile(image_path) and image_path.endswith('.jpg'):
            # Load and process image
            image = Image.open(image_path)
            encoded_image = model.encode_image(image)

            # Generate caption
            caption = model.caption(encoded_image)["caption"]
            captions_dict[i] = caption
            print(f"Caption for {image_file}: {caption}")
    with open(json_filename, 'w') as f:
        json.dump(captions_dict, f, indent=4)


if __name__ == "__main__":
    search_query = "super mario movie trailer"
    search_and_download(search_query)
