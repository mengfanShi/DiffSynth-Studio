from scenedetect import detect, split_video_ffmpeg
from scenedetect import ContentDetector, ThresholdDetector, AdaptiveDetector
import os


def detect_scenes(input_video):
    content_detector = ContentDetector(threshold=30)
    threshold_detector = ThresholdDetector(threshold=15)
    adaptive_detector = AdaptiveDetector(adaptive_threshold=5)

    try:
        scene_list = detect(input_video, content_detector)
    except:
        scene_list = []

    return scene_list


def save_scenes(input_video, output_dir):
    scene_list = detect_scenes(input_video)
    if scene_list:
        split_video_ffmpeg(input_video, scene_list, output_dir=output_dir)
        videos = []
        for file in sorted(os.listdir(output_dir)):
            if file.endswith('mp4'):
                videos.append(os.path.join(output_dir, file))
        return videos
    else:
        return [input_video]
