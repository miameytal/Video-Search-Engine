```markdown
# Video Scene Analyzer & Caption Generator

A Python-based tool that automates the process of searching for videos on YouTube, detecting key scenes, generating descriptive captions using AI, and interactively retrieving and visualizing specific content. The application offers two modes of search:
- **Image Mode:** Uses scene detection and caption generation to allow keyword searches over the generated captions.
- **Video Mode:** Integrates with Google Gemini (or a stub) to search directly within the video and return timestamped scenes.

---

## Features

- **Automated Video Retrieval:** Searches and downloads videos from YouTube based on a user-specified query using `yt_dlp`.
- **Scene Detection & Image Extraction:** Detects key scenes with `SceneDetect` and extracts representative frames using OpenCV.
- **AI-Powered Captioning:** Generates descriptive captions for each scene with a custom machine learning model (via `moondream`) or a dummy stub.
- **Interactive CLI:** Provides a command-line interface with auto-complete and fuzzy search (powered by `prompt_toolkit` and `rapidfuzz`) to quickly locate scenes by keyword.
- **Dynamic Visual Summaries:** Creates collages from selected scenes to offer a visual summary of the content.
- **Flexible Search Modes:** Offers both image-based caption search and direct video scene search via integration with Google Gemini’s API.

---

## Requirements

- **Python 3.7+**
- **Libraries/Packages:**
  - `yt_dlp`
  - `scenedetect`
  - `opencv-python`
  - `Pillow`
  - `moondream` (or your custom ML model package)
  - `rapidfuzz`
  - `prompt_toolkit`
  - `google-generativeai`
  - `requests`
  - `hashlib` (standard library)
  - `json` (standard library)
  - Other standard Python libraries

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *If a `requirements.txt` file is not provided, manually install the packages:*

   ```bash
   pip install yt_dlp scenedetect opencv-python Pillow rapidfuzz prompt_toolkit google-generativeai requests
   ```

---

## Configuration

Set the following environment variables before running the tool:

- **`MOONDREAM_MODEL_PATH`**  
  Path to your AI model file used by `moondream` for caption generation.

- **`GEMINI_API_KEY`**  
  Your API key for accessing Google Gemini’s generative AI services.

- *(Optional)* **`USE_CAPTION_STUB`**  
  Set to `"true"` to use a stub for caption generation instead of the full model.

- *(Optional)* **`USE_GEMINI_STUB`**  
  Set to `"true"` to use a stub for Gemini API calls.

For example, on Unix-like systems:

```bash
export MOONDREAM_MODEL_PATH="/path/to/model"
export GEMINI_API_KEY="your_gemini_api_key"
export USE_CAPTION_STUB="false"
export USE_GEMINI_STUB="false"
```

On Windows, use `set` instead of `export`.

---

## Usage

Run the script from the command line:

```bash
python video_scene_analyzer.py
```

You will be prompted to:

1. **Input a video search query:**  
   e.g., "latest tech review"
2. **Select a search mode:**  
   - Type `"image"` for caption-based search.
   - Type `"video"` for direct video search via Google Gemini.
3. **Follow additional prompts:**  
   - In image mode, enter a keyword to search the generated captions.
   - In video mode, enter a description to retrieve timestamped scenes and generate a collage.

The tool will output messages to the terminal indicating its progress (download status, scene detection, caption generation, etc.) and will display a final collage image if applicable.

---

## How It Works

1. **Search & Download:**  
   The tool uses `yt_dlp` to search YouTube for the query and downloads the top result.

2. **Scene Detection:**  
   Utilizing `SceneDetect` and OpenCV, it processes the video to identify and extract key scene images.

3. **Caption Generation:**  
   Each scene image is processed through an AI model to generate a caption. Captions are saved as a JSON file.

4. **Interactive Search & Collage Creation:**  
   Users can search the captions with fuzzy matching to quickly find scenes, which are then compiled into a visual collage. Alternatively, Google Gemini’s API is used to find scenes directly from the video.

---

## License

This project is open source under the [MIT License](LICENSE).

---

## Acknowledgements

- [yt_dlp](https://github.com/yt-dlp/yt-dlp) for video downloading.
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) for scene detection.
- [Google Generative AI](https://developers.generativeai.google/) for the Gemini API integration.
- [Prompt Toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) for building the interactive CLI.
```
