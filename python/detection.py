from __future__ import division
import torch
import cv2
from torchvision import transforms
from sklearn.cluster import DBSCAN
from torchvision.models import resnet18
import torch.nn as nn
from PIL import Image

device = torch.device("cuda")

# YOLO Object detection model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)

# Load Classification model
classifier = resnet18(pretrained=True)
num_ftrs = classifier.fc.in_features
classifier.fc = nn.Linear(num_ftrs, 3)
classifier.load_state_dict(torch.load('models/model_resnet18.pth'))
classifier.eval()
classifier.to(device)

classes = ['car', 'person', 'rider']

# Trained classification model data transform parameters
mean = [0.43551864, 0.46134753, 0.44724909]
std = [0.22572689, 0.23291896, 0.22396011]

transform_trained = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# performance measurement
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


# Run clustering using DBSCAN algorithm
# points = Nx2 array of coordinates
# eps, min_samples = hyperparameters
# returns dictionary of clusters with key = index, values = list of coordinates in cluster
def run_clustering(points, eps, min_samples):
    print('Running clustering...')
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.fit_predict(points)
    unique_labels = set(labels)
    num_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    print(str(num_clusters) + ' clusters predicted...')
    # add results to dictionary
    clusters = {}
    for l in unique_labels:
        clusters[l] = []
    for i in range(labels.size):
        clusters[labels[i]].append(points[i])
    return clusters


# detect objects in list of images using YOLO model
# set save to True to save images with annotated detections
# returns dataframe of detection results and inference time
def detect_objects(objects, save=False):
    rgb_objects = []
    # convert images to RGB format
    for o in objects:
        rgb_objects.append(cv2.cvtColor(o, cv2.COLOR_BGR2RGB))
    starter.record()  # start GPU timer
    results = model(rgb_objects)  # detect objects
    if save:
        results.save()
    ender.record()  # stop GPU timer
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)  # calculate elapsed time
    return results.pandas().xyxy, curr_time


# run classification of list of images
# objects = list of images
# bbxs = list of corresponding projected bounding box coordinates
# frame = current frame
# f = csv writer to write results to file
# count = count of number of classifications done, used for indexing in results file
def classify(objects, bbxs, frame, f=None, count=None):
    for (o, bb) in zip(objects, bbxs):
        # format input image
        rgb_img = cv2.cvtColor(o, cv2.COLOR_BGR2RGB)  # convert to RGB
        img = Image.fromarray(rgb_img)  # convert to PIL image
        img_t = transform_trained(img)  # transform
        batch = torch.unsqueeze(img_t, 0).to(device)  # unsqueeze and send to GPU

        starter.record()  # start GPU timer
        out = classifier(batch)  # classify image
        ender.record()  # stop GPU timer
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)  # calculate elapsed time

        # Calculate classification results and confidence
        _, indices = torch.sort(out, descending=True)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        if f:
            # write results to file
            min_x = max(0, bb[0])
            max_x = min(4095, bb[2])
            min_y = max(0, bb[1])
            max_y = min(2159, bb[3])
            idx = indices[0][0]
            row = [count, min_x, min_y, max_x, max_y, percentage[idx].item(), idx.cpu().detach().numpy(), classes[idx], frame, curr_time]
            f.writerow(row)
            count += 1
    return count, curr_time


# sends random 10 input into model to warm up gpu for inference time measurement
def warm_gpu():
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)
    for _ in range(10):
        _ = model(dummy_input)
