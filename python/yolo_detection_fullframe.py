import csv

import torch
import cv2
import pandas as pd
import numpy as np
import os

medianFrame = cv2.imread('./utilities/background.jpg')

# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

# Model
device = torch.device("cuda")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)

# performance measurement
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

rgb_data = './data/rgb_video.mp4'


# function to warm up GPU for inference time measurement
def warm_gpu():
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)
    for _ in range(10):
        _ = model(dummy_input)


# run detection
# parameters: num = experiment number
# save_results: if True, saves results to file
# blur: if True and apply_mask is True, blurred mask applied
# if blur=False and apply_mask=True: binary mask applied
def run_detection(num, save_results=True, blur=False, apply_mask=False):
    dc = None
    video_reader = cv2.VideoCapture(rgb_data)
    frame = 0
    inf_times = []
    while True:
        ret, bgr_frame = video_reader.read()

        if not ret: break
        if apply_mask:
            # Convert current frame to grayscale
            gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            # Calculate absolute difference of current frame and
            # the median frame
            dframe = cv2.absdiff(gray_frame, grayMedianFrame)
            # Treshold to binarize
            th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)

            # dilate mask
            kernel = np.ones((15, 15), dtype='uint8')
            mask = cv2.dilate(dframe, kernel, iterations=5)
            if blur:
                # blur mask and apply
                blurred_mask = cv2.blur(mask, [30, 30])
                blurred_image = cv2.blur(bgr_frame, [100, 100])
                blurred_image[blurred_mask == 255] = bgr_frame[blurred_mask == 255]
                rgb_frame = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
            else:
                # apply mask
                masked_dilation = cv2.bitwise_and(bgr_frame, bgr_frame, mask=mask)
                rgb_frame = cv2.cvtColor(masked_dilation, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        # run detection and measure inference time
        starter.record()
        results = model(rgb_frame)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        inf_times.append(curr_time)

        # format results
        df = results.pandas().xyxy[0]
        # results.save() # uncomment to save annotated images
        if not df.empty:
            df['frame'] = frame
            df['time'] = 0
            df['time'][0] = curr_time
            if dc is None:
                dc = df.copy()
            else:
                dc = pd.concat([dc, df], axis=0, ignore_index=True)
        else:
            print('warning, results empty for frame: ', frame)
        print('frame: ' + str(frame))
        frame += 1

    if save_results:
        params = {'function': 'full-frame', 'mask': apply_mask, 'blur': blur}

        # create directory for results
        direct = './results/exp-{}'.format(num)
        if not os.path.isdir(direct):
            os.mkdir(direct)
        # text file containing parameter values
        f = open(os.path.join(direct, 'parameters.txt'), 'w', newline='')
        f.write(str(params))
        dc.to_csv(os.path.join(direct, 'detect.csv'))
    return inf_times


# function to measure inference time over a number of iterations
# exp = experiment number
def inference_time(num_repetitions, exp):
    direct = './results/inference_time'
    if not os.path.isdir(direct):
        os.mkdir(direct)
    f = open(os.path.join(direct, 'fullframe_inference_time_exp-{}.csv'.format(exp)), 'w', newline='')
    writer = csv.writer(f)
    # run main for each interation and write inference time results to file
    warm_gpu()
    for n in range(num_repetitions):
        print('********** RUN {} ************'.format(n))
        times = run_detection(exp, save_results=False)
        writer.writerow(times)
    f.close()


if __name__ == "__main__":
    inference_time(10, 1)
    # run_detection(7)
