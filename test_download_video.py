import unittest
from unittest.mock import patch, MagicMock
from yt_dlp.utils import DownloadError
from download_video import search_and_download, detect_scenes, save_scene_images

class TestDownloadVideo(unittest.TestCase):

    @patch('yt_dlp.YoutubeDL')
    def test_search_and_download_success(self, mock_yt_dlp):
        # Test that the video is downloaded successfully when a valid query is provided
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance
        mock_ydl_instance.extract_info.return_value = {'entries': [{'webpage_url': 'http://example.com', 'title': 'Test Video'}]}
        
        search_and_download("test query")
        
        mock_ydl_instance.download.assert_called_once_with(['http://example.com'])

    @patch('yt_dlp.YoutubeDL')
    def test_search_and_download_no_results(self, mock_yt_dlp):
        # Test that the function handles no search results correctly
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance
        mock_ydl_instance.extract_info.return_value = {'entries': []}
        
        with patch('builtins.print') as mocked_print:
            search_and_download("test query")
            mocked_print.assert_called_once_with("No results found.")

    @patch('yt_dlp.YoutubeDL')
    def test_search_and_download_multiple_results(self, mock_yt_dlp):
        # Test that the function downloads the first result when multiple results are found
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance
        mock_ydl_instance.extract_info.return_value = {'entries': [{'webpage_url': 'http://example.com/1', 'title': 'Test Video 1'}, {'webpage_url': 'http://example.com/2', 'title': 'Test Video 2'}]}
        
        search_and_download("test query")
        
        mock_ydl_instance.download.assert_called_once_with(['http://example.com/1'])

    @patch('yt_dlp.YoutubeDL')
    def test_search_and_download_invalid_query(self, mock_yt_dlp):
        # Test that the function raises a DownloadError for an invalid query
        mock_ydl_instance = MagicMock()
        # Simulate a DownloadError for invalid query
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance
        mock_ydl_instance.extract_info.side_effect = DownloadError("Invalid query")

        with self.assertRaises(DownloadError):
            search_and_download("invalid query")

    @patch('yt_dlp.YoutubeDL')
    def test_search_and_download_empty_query(self, mock_yt_dlp):
        # Test that the function handles an empty query correctly
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance
        mock_ydl_instance.extract_info.return_value = {'entries': []}
        
        with patch('builtins.print') as mocked_print:
            search_and_download("")
            mocked_print.assert_called_once_with("No results found.")

    @patch('yt_dlp.YoutubeDL')
    def test_search_and_download_special_characters(self, mock_yt_dlp):
        # Test that the function handles queries with special characters correctly
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance
        mock_ydl_instance.extract_info.return_value = {'entries': [{'webpage_url': 'http://example.com', 'title': 'Test Video'}]}
        
        search_and_download("!@#$%^&*()")
        
        mock_ydl_instance.download.assert_called_once_with(['http://example.com'])

    @patch('yt_dlp.YoutubeDL')
    def test_search_and_download_long_query(self, mock_yt_dlp):
        # Test that the function handles a very long query correctly
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance
        mock_ydl_instance.extract_info.return_value = {'entries': [{'webpage_url': 'http://example.com', 'title': 'Test Video'}]}
        
        long_query = "a" * 1000
        search_and_download(long_query)
        
        mock_ydl_instance.download.assert_called_once_with(['http://example.com'])

    @patch('yt_dlp.YoutubeDL')
    def test_search_and_download_unicode_query(self, mock_yt_dlp):
        # Test that the function handles queries with unicode characters correctly
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance
        mock_ydl_instance.extract_info.return_value = {'entries': [{'webpage_url': 'http://example.com', 'title': 'Test Video'}]}
        
        search_and_download("测试查询")
        
        mock_ydl_instance.download.assert_called_once_with(['http://example.com'])

    @patch('yt_dlp.YoutubeDL')
    def test_search_and_download_no_internet(self, mock_yt_dlp):
        # Test that the function handles no internet connection
        mock_ydl_instance = MagicMock()
        # Simulate a DownloadError due to no internet connection
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance
        mock_ydl_instance.extract_info.side_effect = DownloadError("No internet connection")

        with self.assertRaises(DownloadError):
            search_and_download("test query")

    @patch('yt_dlp.YoutubeDL')
    def test_search_and_download_partial_results(self, mock_yt_dlp):
        # Test that the function handles None entries in search results
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance
        # Include None entries in the search results
        mock_ydl_instance.extract_info.return_value = {
            'entries': [None, {'webpage_url': 'http://example.com', 'title': 'Test Video'}]
        }

        with patch('builtins.print') as mocked_print:
            search_and_download("test query")
            mocked_print.assert_any_call("Downloading: Test Video")

        mock_ydl_instance.download.assert_called_once_with(['http://example.com'])

    @patch('download_video.detect_scenes')
    @patch('yt_dlp.YoutubeDL')
    def test_detect_scenes_called(self, mock_yt_dlp, mock_detect_scenes):
        # Test that detect_scenes is called after downloading the video
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance
        video_info = {'webpage_url': 'http://example.com', 'title': 'Test Video'}
        mock_ydl_instance.extract_info.return_value = {'entries': [video_info]}
        mock_ydl_instance.prepare_filename.return_value = 'test_video.mp4'

        search_and_download("test query")

        mock_detect_scenes.assert_called_once_with('test_video.mp4')

    @patch('cv2.VideoCapture')
    def test_save_scene_images_invalid_video_path(self, mock_video_capture):
        # Test that save_scene_images handles an invalid video path
        mock_cap_instance = MagicMock()
        mock_video_capture.return_value = mock_cap_instance
        mock_cap_instance.isOpened.return_value = False
        scene_list = [((0, 0), (1, 0))]

        with patch('builtins.print') as mocked_print:
            save_scene_images('invalid_video.mp4', scene_list)
            mocked_print.assert_called_with('Could not open video file: invalid_video.mp4')

        mock_cap_instance.isOpened.assert_called_once()
        mock_cap_instance.release.assert_called_once()
        mock_cap_instance.set.assert_not_called()
        mock_cap_instance.read.assert_not_called()

if __name__ == "__main__":
    unittest.main()
