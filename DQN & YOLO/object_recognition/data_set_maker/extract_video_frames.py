import cv2
import os
import multiprocessing

def extract_frames(video_path, output_path):
    video = cv2.VideoCapture(video_path)
    count = 0
    fps = int(video.get(cv2.CAP_PROP_FPS))
    saved_count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if count % 10 == 0:
            cv2.imwrite(os.path.join(output_path, f'{os.path.basename(video_path)}_{saved_count}.jpg'), frame)
            saved_count += 1

        count += 1

    video.release()

if __name__ == '__main__':
    vids_path = 'D:\\Osu!_object_detection_dataset\\vids\\'
    output_path = 'D:\\Osu!_object_detection_dataset\\frames\\'
    num_processes = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(num_processes)
    for replay in os.scandir(vids_path):
        replay_name = replay.name
        pool.apply_async(extract_frames, args=(replay.path, output_path))

    pool.close()
    pool.join()
