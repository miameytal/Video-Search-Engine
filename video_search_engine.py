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
from prompt_toolkit.styles import Style
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.formatted_text import HTML
import requests
import google.generativeai as genai
import json

# Retrieve the model path from an environmental variable
model_path = os.getenv('MOONDREAM_MODEL_PATH')
model = md.vl(model=model_path)

# Define styles for prompt_toolkit
style = Style.from_dict({
    'prompt': 'ansicyan bold',
    'info': 'ansigreen',
    'warning': 'ansiyellow',
    'error': 'ansired bold',
})

def search_and_download(query):
    # Create a unique filename for the JSON file based on the query
    query_hash = hashlib.md5(query.encode()).hexdigest()

    # Determine the video filepath based on the query hash
    video_filename = f"{query_hash}.mp4"
    video_filepath = os.path.join(os.getcwd(), video_filename)

    if os.path.exists(video_filepath):
        print_formatted_text(HTML(f'<info>{video_filename} already exists; skipping download.</info>'), style=style)
        return video_filepath

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
                print_formatted_text(HTML('<warning>No results found.</warning>'), style=style)
                return None

            # Get the first valid result
            video_info = search_results[0]
            video_url = video_info['webpage_url']

            print_formatted_text(HTML(f'<info>Downloading: {video_info["title"]}</info>'), style=style)

            # Download the video
            ydl.download([video_url])

            # Get the filename of the downloaded video
            downloaded_video_filename = ydl.prepare_filename(video_info)
            print_formatted_text(HTML('<info>Video downloaded</info>'), style=style)

            # Rename the downloaded video to include the hash
            os.rename(downloaded_video_filename, video_filename)

    except DownloadError as e:
        print_formatted_text(HTML(f'<error>An error occurred: {e}</error>'), style=style)
        raise
    except Exception as e:
        print_formatted_text(HTML(f'<error>An unexpected error occurred: {e}</error>'), style=style)

    return video_filepath

def detect_scenes(video_path, json_filename):
    # Create a directory to save scene images
    output_dir = f"scene_images_{hashlib.md5(video_path.encode()).hexdigest()}"
    os.makedirs(output_dir, exist_ok=True)

    # Check if scene images for this video already exist
    scene_images_exist = any(fname.endswith('.jpg') for fname in os.listdir(output_dir))
    if scene_images_exist:
        print_formatted_text(HTML('<info>Scene images for this video already exist; skipping scene detection.</info>'), style=style)
        return

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

        # Save images of detected scenes
        save_scene_images(video_path, scene_list, json_filename, output_dir, scene_images_exist)

    finally:
        # Release the video manager
        video_manager.release()
    
    return scene_list

def save_scene_images(video_path, scene_list, json_filename, output_dir, scene_images_exist):
    # Check if scene images for this video already exist
    if scene_images_exist:
        print_formatted_text(HTML('<info>Scene images for this video already exist; skipping saving scene images.</info>'), style=style)
        return

    # Determine the number of digits required to represent the scene numbers
    num_scenes = len(scene_list)
    num_digits = len(str(num_scenes))

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print_formatted_text(HTML(f'<error>Could not open video file, {video_path}</error>'), style=style)
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
        else:
            print_formatted_text(HTML(f'<warning>Failed to read frame at scene {i+1}</warning>'), style=style)
    cap.release()

    # Generate captions for the saved scene images
    generate_caption(output_dir, json_filename)

def generate_caption(output_dir, json_filename):
    # Check if captions already exist
    if os.path.exists(json_filename):
        print_formatted_text(HTML('<info>Captions already exist; skipping caption generation.</info>'), style=style)
        return

    import json
    if os.getenv('USE_CAPTION_STUB') == 'true': #TODO: remove this stub
        # Produce dummy captions for each image in the directory
        images = [f for f in sorted(os.listdir(output_dir)) if f.endswith('.jpg')]
        captions_dict = {}
        for i, image_file in enumerate(images, start=1):
            captions_dict[i] = f"Dummy caption {i}"

        with open(json_filename, 'w') as f:
            json.dump(captions_dict, f, indent=4)
        print_formatted_text(HTML('<info>Dummy captions generated. JSON file created.</info>'), style=style)
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
    search_word = prompt(HTML('<prompt>Search the video using a word: </prompt>'), completer=completer, style=style).strip().lower()

    # Find and print the scenes that contain the search word using rapidfuzz
    found_scenes = [scene for scene, caption in captions_dict.items() if process.extractOne(search_word, [caption.lower()], score_cutoff=50)]
    if found_scenes:
        create_collage(found_scenes, len(captions_dict))
    else:
        print_formatted_text(HTML(f'<warning>No scenes found containing the word "{search_word}".</warning>'), style=style)


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
    print_formatted_text(HTML('<info>Collage saved as "collage.png".</info>'), style=style)

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
        print_formatted_text(HTML('<info>Uploading file...</info>'), style=style)
        video_file = genai.upload_file(path=video_file_name, display_name=display_name, resumable=True)
        print_formatted_text(HTML('<info>Completed upload.</info>'), style=style)
    else:
        print_formatted_text(HTML(f'<info>File URI: {video_file.uri}</info>'), style=style)

    # Check the state of the uploaded file.
    while video_file.state.name == "PROCESSING":
        print(".", end="")
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)

    # Build a prompt that requests a JSON object with a "timestamps" array
    query_prompt = (
        f"Search through this video for scenes related to the description '{word}'.\n"
        "Return your answer in valid JSON with a top-level 'timestamps' key containing an array of timestamps in the format hh:mm:ss (hours are optional for videos under an hour).\n"
        "Every timestamp must be less than the length of the video itself.\n"
        "If no scenes match, provide an empty array for 'timestamps'.\n"
        "Example response (assuming the video is 100 minutes long):\n"
        "```json\n"
        "{\n"
        "  \"timestamps\": [\"00:02:34\", \"00:05:40\", \"00:08:50\"]\n"
        "}\n"
        "```\n"
        "Another example response (assuming the video is 10 minutes long):\n"
        "```json\n"
        "{\n"
        "  \"timestamps\": [\"02:34\", \"05:40\", \"08:50\"]\n"
        "}\n"
        "```"
    )
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([video_file, query_prompt])
    response_json = response.to_dict()
    
    # Extract the JSON string from the response and parse it
    content = response_json['candidates'][0]['content']['parts'][0]['text']
    timestamps_json = json.loads(content.strip('```json\n').strip('\n```'))
    timestamps = timestamps_json.get("timestamps", [])
    return timestamps

def call_gemini_stub(video_file_name, word):
    """
    Stub function to mimic the behavior of the Google Gemini model.
    Returns a mock response with timestamps.
    """
    print_formatted_text(HTML('<info>Using Gemini stub...</info>'), style=style)
    # Mock response with timestamps in hh:mm:ss format
    mock_response = {
        "timestamps": ["00:02:34", "00:05:40", "00:08:50"]
    }
    return mock_response["timestamps"]

def parse_timestamp(timestamp_str):
    """
    Parse a timestamp string in the format hh:mm:ss or mm:ss and return the time in milliseconds.
    """
    parts = timestamp_str.split(':')
    if len(parts) == 2:
        # Format mm:ss
        minutes, seconds = map(float, parts)
        return int((minutes * 60 + seconds) * 1000)
    elif len(parts) == 3:
        # Format hh:mm:ss
        hours, minutes, seconds = map(float, parts)
        return int((hours * 3600 + minutes * 60 + seconds) * 1000)
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}")

def create_collage_from_timestamps(video_file_name, timestamps):
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_file_name)
    if not cap.isOpened():
        print_formatted_text(HTML(f'<error>Could not open video file, {video_file_name}</error>'), style=style)
        cap.release()
        return

    images = []
    for timestamp in timestamps:
        # Convert timestamp to milliseconds and add a buffer of 500 milliseconds
        timestamp_ms = parse_timestamp(timestamp) + 500
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        ret, frame = cap.read()
        if ret:
            # Convert frame to PIL image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            images.append(image)
        else:
            print_formatted_text(HTML(f'<warning>Failed to read frame at timestamp {timestamp}</warning>'), style=style)

    cap.release()

    if images:
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
        print_formatted_text(HTML('<info>Collage saved as "collage.png".</info>'), style=style)

        # Display the collage
        collage.show()
    else:
        print_formatted_text(HTML('<warning>No images to create a collage.</warning>'), style=style)

if __name__ == "__main__":

    # Ask the user to input the search query
    search_query = prompt(HTML('<prompt>What video do you want to search?: </prompt>'), style=style).strip()

    # Ask the user to choose between image model and video model
    model_choice = prompt(HTML('<prompt>Choose a model to search the video (image/video): </prompt>'), style=style).strip().lower()

    video_filepath = search_and_download(search_query)
    
    if video_filepath:
        if model_choice == "image":
            # Generate and search captions for a word using the image model
            json_filename = f'scene_captions_{hashlib.md5(search_query.encode()).hexdigest()}.json'
            detect_scenes(video_filepath, json_filename)
            search_captions(json_filename)
        elif model_choice == "video":
            # Ask the user what to find in the video using the video model
            search_word = prompt(HTML('<prompt>Using a video model. What would you like me to find in the video? </prompt>'), style=style).strip()
            # Use the stub function if the environment variable is set, otherwise use the actual API call
            if os.getenv('USE_GEMINI_STUB') == 'true':
                timestamps = call_gemini_stub(video_filepath, search_word)
            else:
                timestamps = call_gemini(video_filepath, search_word)
            # Create and display a collage of scenes at the given timestamps
            create_collage_from_timestamps(video_filepath, timestamps)
        else:
            print_formatted_text(HTML('<error>Invalid choice. Please choose either "image" or "video".</error>'), style=style)
