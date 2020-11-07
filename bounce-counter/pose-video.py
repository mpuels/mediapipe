import itertools
import logging
from pathlib import Path
from typing import Iterable

import cv2

import mediapipe as mp

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

INPUT_VIDEO = Path('/home/mpuels/projis/bounce-counter/data/videos/hand-bounce-001.mp4')
OUTPUT_DIR = Path('/home/mpuels/projis/bounce-counter/data/images')
OUTPUT_DIR_IMAGES_WITH_LANDMARKS = Path('/home/mpuels/projis/bounce-counter/data/images-with-landmarks')

CONVERT_VIDEO_TO_IMAGES = False
ONLY_COMPUTE_LANDMARKS_FOR_FIRST_N_IMAGES = False


def main():
    if CONVERT_VIDEO_TO_IMAGES:
        convert_video_to_images(INPUT_VIDEO, OUTPUT_DIR)
    input_video_name = INPUT_VIDEO.stem
    file_list = sorted(OUTPUT_DIR.glob(f'{input_video_name}-*'))

    if ONLY_COMPUTE_LANDMARKS_FOR_FIRST_N_IMAGES:
        file_list = itertools.islice(file_list, 0, 10)

    write_landmarks(file_list, OUTPUT_DIR_IMAGES_WITH_LANDMARKS)


def convert_video_to_images(input_video: Path, output_dir: Path):
    cap = cv2.VideoCapture(str(input_video))
    video_name = input_video.stem
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output_filepath = output_dir / f'{video_name}-{i:05}.jpg'
        cv2.imwrite(str(output_filepath), frame)
        i = i + 1

    cap.release()
    cv2.destroyAllWindows()


def write_landmarks(file_list: Iterable[Path], output_dir: Path):
    # For static images:
    pose = mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5)
    for filepath in file_list:
        logger.info(f'write landmark for {filepath}')
        image = cv2.imread(str(filepath))
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw pose landmarks on the image.
        if results.pose_landmarks is None:
            logger.warning(f'no landmarks found in image: {filepath}')
            continue

        logger.debug(f'nose landmark: {results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]}')
        annotated_image = image.copy()

        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS)

        output_filepath = output_dir / f'{filepath.stem}.png'
        cv2.imwrite(str(output_filepath), annotated_image)
    pose.close()


if __name__ == '__main__':
    main()
