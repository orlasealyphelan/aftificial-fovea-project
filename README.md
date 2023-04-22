# Artificial Fovea Project
This repository contains code used in the implementation of my Master's project 'Implementing an ‘Artificial Fovea’ for more efficient monitoring using the sensor fusion of an event-based and frame-based camera'
With example results.

## Directories
### matlab
MATLAB code and images used to investigate calibration of cameras and projection of points between image planes.  

`8.9` & `ED`: Calibration images used in MATLAB Stereo Camera Calibrator.  
`stereoParams.mat`: Calibration results.  
`projectoinFunctions.m`: Projects points from one camera image plane to the other using build in MATLAB functions.  
`projectoinEquation.m`: Projects points from one camera image plane to the other using derived equation with exact extrinsic parameters for each pattern.  
`calculateAverageKRt.m`: Calculated average camera parameters for use in projection - `av_divKRt`= (K1/K2)*([R1 t1]/[R2 t2])average.  
`av_divKRt.mat`: File containing `av_divKRt` calculated in `calculateAverageKRt`.  
`av_KRt_projection.m`: Script to project points using equation with average extrinsic parameters.  
`calculateError.m`: Funtion file to calculate the reprojection error.  

### python
Python code used in main implementation, performance measurement, and classifier training.  

1. `main.py`: main python implementation.  
**Functions**:  
- `main`: run object detection, specifying function paramter as:  
  - 'projection': Visualise results of clustering and projection of bounding boxes. 
  - 'classify': Run full implementation using classification model.
  - 'OD': Run full implementation using YOLOv5s object detection model.
  - 'ODMask': Run full implementation using YOLOv5s object detection model and apply background mask. Set blur=True to use blurred mask over binary mask.

- `measure_inference_time`: run full implementation `num_repetitions` times and measure inference times every frame. `function` same as above.  

**Other files used in main implementation**
- `detection.py`: Functions related to object detection & classification.
- `points.py`: Functions related to manipulation of points.
- `tools.py`: Functions for various other tools needed in implementation.

2. `convert_results.py`: convert results from `main` to format used in performance measurement.  
- `prediction` = True for predicted results, False for ground truth annotations
- `classify` = True/Fasle for classication/object detection results
- `exp_num`: experiment number of results.  

3. `measure_performance.py`: reads in formatted inputs, calculates:
- `read_inputs`: reads in formatted annotations and predictions and returns list used in performance measurement.
- `coco_map`: calculates following metrics for each class over a range of specified intersection over union thresholds and writes resuls to `map.csv`:
  - Average Precision (mAP)
  - Precision
  - Recall
  - True Positives (TP)
  - True Negatives (FP)
- `calculate_iou`: calculates intersection over union of each detection and writes results to `iou.csv`

`mean_average_precision.py` & `iou.py`: taken from [this repository](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/metrics) with some minor adjustments made to `mean_average_precision` to return all metrics for each rather than just mean average precision.  

4. `yolo_detection_fullframe.py`: YOLOv5s run on each full frame of the RGB video. Used for performance comparison.  

#### Sub-directories
- `annotations`: annotations of RGB video
  - `bike_person`: used for YOLO, bicycle and person annotated seperately.  
  - `rider`: used with classifier, bicycle class = rider with person included.  
- `data`: RGB and event data.  
- `models`: `model_resnet18.pth` - trained classification model.  
- `utilities`: 
  - `median_filter.py`: generates background image using median filter.  
- `results`: some example results. In each experiment folder:
  - `detect.csv`/`classify.csv`: Object detection/classification results.
  - `parameters.txt`: List of parameters used in experiment.
  - `formatted.csv`: Formatted results from running `convert_results.py`.
  -`iou.csv` & `map.csv`: Performance evaluation from running `measure_performance.py`.  
  
### training
`dataAquisition`: Extract data from BDD dataset and collected data for classifier training.
1. Download [BDD dataset](https://bdd-data.berkeley.edu/) and add directory location to `dataAquisition/BDD_extract.py`.  
2. Run `dataAquisition/BDD_extract.py` and `dataAquisition/DanganData.py` to extract data.  

**Before Training**
Split up data into train, validation, and test splits and strucutre data according to structure below and run `train_classifier.py`.  
![image](https://user-images.githubusercontent.com/130498225/233446865-6bd04d87-45f8-4762-9e26-8f9d926f6d73.png)

`train_classifier.py` taken from [tutorial](https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py) and adapted to use mean and standard deviation of dataset specified, and to plot accuracy and loss plots. Some visualisations removed from code also.

