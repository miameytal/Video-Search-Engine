import yt_dlp
from yt_dlp.utils import DownloadError
import time
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import cv2
import os
import moondream as md
from PIL import Image, ImageDraw
import hashlib
from rapidfuzz import process
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
import requests
import google.generativeai as genai

# Retrieve the model path from an environmental variable
model_path = os.getenv('MOONDREAM_MODEL_PATH')
model = md.vl(model=model_path)

def search_and_download(query):
    # Create a unique filename for the JSON file based on the query
    query_hash = hashlib.md5(query.encode()).hexdigest()
    json_filename = f'scene_captions_{query_hash}.json'

    if os.path.exists(json_filename):
        print(f"{json_filename} already exists; skipping download and caption generation.")
        return json_filename

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
                return None

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

    return json_filename

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
    
    return scene_list

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

def search_captions(json_filename):
    import json
    # Load the captions from the JSON file
    with open(json_filename, 'r') as f:
        captions_dict = json.load(f)

    # Extract unique words from captions for auto-complete
    words = set()
    for caption in captions_dict.values():
        words.update(caption.lower().split())
    completer = WordCompleter(list(words), ignore_case=True)

    # Prompt the user to enter a word to search for with auto-complete
    search_word = prompt("Search the video using a word: ", completer=completer).strip().lower()

    # Find and print the scenes that contain the search word using rapidfuzz
    found_scenes = [scene for scene, caption in captions_dict.items() if process.extractOne(search_word, [caption.lower()], score_cutoff=50)]
    if found_scenes:
        print(f"Scenes containing the word '{search_word}': {found_scenes}")
        create_collage(found_scenes, len(captions_dict))
    else:
        print(f"No scenes found containing the word '{search_word}'.")


def create_collage(found_scenes, total_scenes):
    # Calculate the number of digits required for zero-padding
    num_digits = len(str(total_scenes))

    # Load the images for the found scenes
    images = [Image.open(f"scene_images/scene_{str(scene).zfill(num_digits)}.jpg") for scene in found_scenes]

    # Determine the size of the collage
    num_images = len(images)
    collage_size = int(num_images**0.5) + (1 if int(num_images**0.5)**2 < num_images else 0)
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a new blank image for the collage
    collage = Image.new('RGB', (collage_size * max_width, collage_size * max_height))

    # Paste the images into the collage
    for idx, img in enumerate(images):
        x_offset = (idx % collage_size) * max_width
        y_offset = (idx // collage_size) * max_height
        collage.paste(img, (x_offset, y_offset))

    # Save the collage to a file
    collage.save('collage.png')
    print("Collage saved as 'collage.png'.")

    # Display the collage
    collage.show()

def call_gemini(video_file_name, word):
    """
    Configures the API key from an environment variable and makes a request to the Google Gemini model
    to search through a single video for scenes related to the given word.
    Returns the timestamps of the scenes where the word was detected.
    """
    # Extract the display name from the video file name
    display_name = os.path.splitext(os.path.basename(video_file_name))[0]

    api_key = os.getenv("GEMINI_API_KEY", "")
    genai.configure(api_key=api_key)

    # Get file list in Gemini
    fileList = genai.list_files(page_size=100)

    # Check uploaded file.
    video_file = next((f for f in fileList if f.display_name == display_name), None)
    if video_file is None:
        print(f"Uploading file...")
        video_file = genai.upload_file(path=video_file_name, display_name=display_name, resumable=True)
        print(f"Completed upload: {video_file.uri}")
    else:
        print(f"File URI: {video_file.uri}")

    # Check the state of the uploaded file.
    while video_file.state.name == "PROCESSING":
        print(".", end="")
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)

    # Build a prompt that requests a JSON object with a "timestamps" array
    query_prompt = (
        f"Search through this video for scenes related to the word '{word}'.\n"
        "Return your answer in valid JSON with a top-level 'timestamps' key containing an array of timestamps.\n"
        "If no scenes match, provide an empty array for 'timestamps'."
    )
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([video_file, query_prompt])
    response_json = response.to_dict()
    
    print("Response JSON:", response_json)
    
    timestamps = response_json.get("timestamps", [])
    return timestamps

if __name__ == "__main__":
    search_query = "super mario movie trailer"
    json_filename = search_and_download(search_query)
    
    if json_filename:
        # Search the captions for a word
        search_captions(json_filename)


    call_gemini(r"C:\Users\User\Desktop\Mia\SDAI\Ex_2.2\Video-Search-Engine\The Super Mario Bros. Movie ｜ Official Trailer [TnGl01FkMMo].mp4", "fire")

