import pandas as pd
from detection import *
from points import *
from tools import *

# Locations of event camera and rgb camera data
event_data = './data/event_data.csv'
rgb_data = './data/rgb_video.mp4'

# Background frame used for median filtering
medianFrame = cv2.imread('./utilities/background.jpg')
# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

# Clustering Hyperparameters
eps = 20
min_samples = 10

# pixel error used for cropping
pixel_error = 100


# function = 'projection'/'classify'/'OD'/'ODMask'
# num = experiment number for saving results
# curr_epoch = starting epoch of event camera data in microseconds
# accumulation_time = accumulation time for event frame generation
# save_results - set to True to save results
# inference_time - set to True to record inference times
# blur - set to True if blurred mask used in 'ODMask' function
def main(function, num, curr_epoch=87033333, accumulation_time=33333, save_results=False, inference_time=False, blur=False):
    inf_times = []
    trained_count = 0
    current_frame = 0
    slice_list = []
    boxes = []
    df = None
    projection_folder = None

    # data readers to read in data from each camera
    video_reader = cv2.VideoCapture(rgb_data)
    csv_reader = pd.read_csv(event_data, delimiter=';', chunksize=5000)

    if save_results:
        params = {'curr_epoch': curr_epoch, 'eps': eps, 'min_samples': min_samples, 'function': function,
                  'blur': blur, 'pixel_error': pixel_error}

        # create directory for results
        direct = './results/exp-{}'.format(num)
        if not os.path.isdir(direct):
            os.mkdir(direct)
        # text file containing parameter values
        f = open(os.path.join(direct, 'parameters.txt'), 'w', newline='')
        f.write(str(params))

        # set up results files for each function
        if function == 'classify':
            f_c = open(os.path.join(direct, 'classify.csv'), 'w', newline='')
            writer = csv.writer(f_c)
            writer.writerow(['', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name', 'frame', 'time'])
        elif function.startswith("OD"):
            df = None
        elif function == 'projection':
            projection_folder = os.path.join(direct, 'projections')
            if not os.path.isdir(projection_folder):
                os.mkdir(projection_folder)
        else:
            writer = None
    else:
        writer = None

    # loop over event camera csv file
    for chunk in csv_reader:
        array = chunk.to_numpy()  # convert data to array

        for row in array:
            slice_list.append(row)  # add row to list
            # check if frame accumulation time is reached for further processing
            if row[0] >= curr_epoch + accumulation_time:
                print('Frame: ' + str(current_frame))

                # convert to array and generate event frame
                slice_arr = np.array(slice_list)
                event_frame = generate_event_frame(slice_arr, current_frame)

                # crop event frame to match aspect ratio of rgb frame
                event_frame = event_frame[80: 418, 0:640]

                # read in rgb frame and downscale to match resolution of event frame
                ret, rgb_frame = video_reader.read()
                resized_rgb = cv2.resize(rgb_frame, (640, 338), interpolation=cv2.INTER_AREA)

                # undistort images
                event_undistort = cv2.undistort(event_frame, cameraMatrix_2, dist_2)
                resized_rgb = cv2.undistort(resized_rgb, cameraMatrix_1, dist_1)

                # format points and run clustering
                formatted_points = format_points(slice_arr)
                clusters = run_clustering(formatted_points, eps, min_samples)

                # loop over clusters and calculate bounding boxes for each
                for key in clusters:
                    if key != -1:
                        coord_array = np.array(clusters[key])
                        bounding_coords = bounding_coordinates(coord_array)
                        boxes.append(bounding_coords)

                # draw bounding boxes on event frame
                event_image = draw_rectangles(event_undistort, boxes, (0, 0, 255))

                objects_resized = []  # list of bounding boxes in resized rgb image
                objects = []  # list of bounding boxes in original rgb image
                for box in boxes:
                    # project bounding box coordinates to rgb camera image plane
                    pp = translate([(box[0], box[1]), (box[0], box[3]), (box[2], box[3]), (box[2], box[1])])
                    objects_resized.append(pp)
                    objects.append(convert_back(pp))  # convert points to original rgb image resolution

                # remove out of frame points and draw bounding boxes
                objects_resized = remove_oof_points(objects_resized)
                rgb_image = draw_rectangles(resized_rgb, objects_resized, (255, 0, 0))

                match function:
                    case "projection":
                        # visualize projection results
                        disp_side_by_side(rgb_image, event_image, current_frame, projection_folder)
                    case "classify":
                        # crop images using projected bounding boxes
                        # with lower pixel error for classification
                        cropped_regions = crop_frame(objects, rgb_frame, pixel_error=25)
                        warm_gpu()  # warm up gpu for inference time measurement
                        trained_count, time = classify(cropped_regions, objects, current_frame, writer, trained_count)
                        # save inference time for frame
                        if inference_time:
                            inf_times.append(time)
                    case "OD":
                        # crop images using projected bounding boxes
                        cropped_regions = crop_frame(objects, rgb_frame)
                        warm_gpu()
                        results, time = detect_objects(cropped_regions)
                        if inference_time:
                            inf_times.append(time)
                        if save_results:
                            # format results
                            df = format_results(results, objects, df, current_frame, time)
                    case "ODMask":
                        # Convert current frame to grayscale
                        gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
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
                            masked = cv2.blur(rgb_frame, [100, 100])
                            masked[blurred_mask == 255] = rgb_frame[blurred_mask == 255]
                        else:
                            # apply mask
                            masked = cv2.bitwise_and(rgb_frame, rgb_frame, mask=mask)

                        # crop masked results and detect objects
                        cropped_regions = crop_frame(objects, masked)
                        warm_gpu()
                        results, time = detect_objects(cropped_regions)
                        if save_results:
                            # format results
                            df = format_results(results, objects, df, current_frame, time)
                current_frame += 1  # increment frame
                curr_epoch = row[0]  # update current epoch
                slice_list.clear()  # clear list of event coordinates
                boxes.clear()  # clear bounding boxes list

    video_reader.release()
    cv2.destroyAllWindows()

    # close results files and write object detection results into csv file
    if save_results:
        f.close()
        if df is not None:
            df.to_csv(os.path.join(direct, 'detect.csv'))
    # return inference times
    return inf_times


# formats object detection results for evaluation
def format_results(results, objects, df, current_frame, time):
    flag = True
    for (r, box) in zip(results, objects):
        min_x = max(0, box[0] - pixel_error)
        min_y = max(0, box[1] - pixel_error)
        if not r.empty:
            r['frame'] = current_frame
            r['time'] = 0
            if flag:
                r['time'][0] = time
                flag = False
            # update bounding box to match full frame image
            for ind in r.index:
                r['xmin'][ind] += min_x
                r['ymin'][ind] += min_y
                r['xmax'][ind] += min_x
                r['ymax'][ind] += min_y
            if df is None:
                df = r.copy()
            else:
                # combine with results from previous frame
                df = pd.concat([df, r], axis=0, ignore_index=True)
    return df


# function to measure inference time over a number of iterations
# function = 'OD'/'ODMask'/'classify'
# exp = experiment number
def measure_inference_time(function, exp, num_repetitions):
    direct = './results/inference_time'
    if not os.path.isdir(direct):
        os.mkdir(direct)
    f = open(os.path.join(direct, '{}_inference_time_exp-{}.csv'.format(function, exp)), 'w', newline='')
    writer = csv.writer(f)
    # run main for each interation and write inference time results to file
    for n in range(num_repetitions):
        print('********** RUN {} ************'.format(n))
        times = main(function, exp, save_results=False, inference_time=True)
        writer.writerow(times)


if __name__ == "__main__":
    # measure_inference_time('OD', 3, 2)
    main('ODMask', 5, save_results=True, blur=True)
