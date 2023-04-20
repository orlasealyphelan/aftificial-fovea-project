import json
import os
import cv2

classes = {'person': 0, 'car': 0, 'rider': 0}  # dictionary to count instances for each label
results_dir = 'BDD_crop'

# create directories for images
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)
for key in classes:
    if not os.path.isdir(os.path.join(results_dir, key)):
        os.mkdir((os.path.join(results_dir, key)))

# read in images and annotations
data_dir = '<path-to-dataset>'
img_dir = os.path.join(data_dir, '/bdd100k_images_100k/images/100k/train')
label_file = os.path.join(data_dir, '/bdd100k_images_100k/bdd100k/labels/bdd100k_labels_images_train.json')
json_file = open(label_file)
data = json.load(json_file)

# count and flag used for processing
image = 0
flag = True

# loop over json annotations file
for img in data:
    crop_count = 0
    name = img['name']  # image name
    for box in img['labels']:
        category = box['category']  # class name
        # check for target classes and number of obtained images < 2000
        if category in list(classes.keys()) and classes[category] < 2000:
            # determine bounding box coordiantes
            bbox = box['box2d']
            xmin = round(bbox['x1'])
            xmax = round(bbox['x2'])
            ymin = round(bbox['y1'])
            ymax = round(bbox['y2'])
            # check size of object
            if (xmax-xmin) > 50 and (ymax-ymin) > 50:
                classes[category] += 1  # increment counter
                directory = './BDD_crop/' + category  # directory to save results
                orginal_img = cv2.imread(os.path.join(img_dir, name))  # read in image
                shape = orginal_img.shape
                # crop image
                crop = orginal_img[max(0, ymin-10):min(shape[0], ymax+10), max(0, xmin-10):min(shape[1], xmax+10)]
                # save image
                if os.path.isdir(directory):
                    cv2.imwrite(os.path.join(directory, 'image' + str(image) + 'c' + str(crop_count) + '.jpg'), crop)
                    print('Saving image to: ', directory)
                crop_count += 1
    image += 1

print(classes)
