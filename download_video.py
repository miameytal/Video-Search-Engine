import yt_dlp
from yt_dlp.utils import DownloadError

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
    except DownloadError as e:
        print(f"An error occurred: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    search_query = "super mario movie trailer"
    search_and_download(search_query)
