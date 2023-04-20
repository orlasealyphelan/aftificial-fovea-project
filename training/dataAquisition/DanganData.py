import torch
import cv2
from PIL import Image
import os

# Model
device = torch.device("cuda")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)


def detect_and_crop(data, direct, labels):
    # create directories for results
    if not os.path.isdir(direct):
        os.mkdir(direct)
    for label in labels:
        if not os.path.isdir(os.path.join(direct, label)):
            os.mkdir((os.path.join(direct, label)))

    # read in viddeo
    video_reader = cv2.VideoCapture(data)
    frame = 0

    while True:
        ret, bgr_frame = video_reader.read()

        if not ret: break
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)  # convert to RGB
        results = model(rgb_frame)  # detect objects
        df = results.pandas().xyxy[0]  # format results

        # loop over detections
        num_detections = len(df.index)
        for i in range(num_detections):
            class_id = df['class'][i]
            if class_id < 3:
                # get class label and bounding box coordinates of target classes
                label = labels[class_id]
                directory = os.path.join(direct, label)
                xmin = round(df['xmin'][i])
                xmax = round(df['xmax'][i])
                ymin = round(df['ymin'][i])
                ymax = round(df['ymax'][i])

                # change height of rider bounding box
                if label == 'rider':
                    ymin -= round((ymax-ymin)/2)
                    ymin = max(0, ymin)
                    print('bicycle detected, adjusting height')
                # crop image and save if target class
                crop = rgb_frame[ymin:ymax, xmin:xmax]
                img = Image.fromarray(crop, "RGB")
                if os.path.isdir(directory):
                    img.save(f"{directory}/f{frame}c{i}.jpg")
            else:
                print(f'{label} detected, image not saved')
        print('frame: ' + str(frame))
        frame += 1


if __name__ == '__main__':
    rgb_data = '../../python/data/rgb_video.mp4'
    direct = 'DanganData/'
    labels = ['person', 'rider', 'car']
    detect_and_crop(rgb_data, direct, labels)
