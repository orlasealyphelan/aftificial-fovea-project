from mean_average_precision import *
import numpy as np
import csv

# location of ground truth annotation files
# rider file used for classifier
# bike_person used for YOLO
gt_files = {'rider': './annotations/rider/formatted_gt.csv',
            'bike_person': './annotations/bike_person/formatted_gt.csv'}


# converts tensor to numpy
def tensor2val(tensor):
    return tensor.cpu().detach().numpy().item()


# calculates IoU between bounding boxes
# results, gt = lists of rows of formatted results
# fname = output csv filename
def calculate_iou(results, gt, fname):
    fout = open(fname, 'w', newline='')
    writer = csv.writer(fout)
    writer.writerow(['frame', 'class', 'iou', 'x1', 'y1', 'x2', 'y2'])
    for result in results:
        best_iou = 0
        for bb in gt:
            if bb[0] == result[0]:  # check frames match
                iou = intersection_over_union(
                    torch.tensor(result[3:]),
                    torch.tensor(bb[3:]),
                    box_format='corners',
                )
                if iou > best_iou:
                    best_iou = iou  # update iou
            elif bb[0] > result[0]:
                break  # break loop and move on to next frame
        # replace confidence (row[2]) with best iou and write to file
        row = result.copy()
        if best_iou > 0:
            row[2] = tensor2val(best_iou)
        else:
            row[2] = 0
        writer.writerow(row)
    fout.close()


# calculates map, precision, recall, True Positive, and False Positives over a range of IoU tresholds
# calculate mAP @ start:stop:inc
# results, gt = lists of rows of formatted results
# fname = output csv filename
def coco_map(results, gt, start, stop, inc, fname):
    fout = open(fname, 'w', newline='')
    writer = csv.writer(fout)
    writer.writerow(['treshold', 'class', 'mAP', 'precision', 'recall', 'TP', 'FP'])
    thresh = np.arange(start, stop + inc, inc)
    map_accum = 0
    for t in thresh:
        print('threshold=', t)
        maps, precisions, recalls, TPs, FPs = mean_average_precision(results, gt, iou_threshold=t)
        car_results = {
            'map': tensor2val(maps[0]),
            'precision': tensor2val(precisions[0][-1]),
            'recall': tensor2val(recalls[0][-1]),
            'TP': tensor2val(TPs[0]) if len(TPs) > 0 else 'null',
            'FP': tensor2val(FPs[0]) if len(FPs) > 0 else 'null'
        }
        person_results = {
            'map': tensor2val(maps[1]),
            'precision': tensor2val(precisions[1][-1]),
            'recall': tensor2val(recalls[1][-1]),
            'TP': tensor2val(TPs[1]) if len(TPs) > 0 else 'null',
            'FP': tensor2val(FPs[1]) if len(FPs) > 0 else 'null'
        }
        rider_results = {
            'map': tensor2val(maps[2]),
            'precision': tensor2val(precisions[2][-1]),
            'recall': tensor2val(recalls[2][-1]),
            'TP': tensor2val(TPs[2]) if len(TPs) > 0 else 'null',
            'FP': tensor2val(FPs[2]) if len(FPs) > 0 else 'null'
        }
        average_results = {
            'map': tensor2val(sum(maps) / len(maps)),
            'precision': tensor2val((precisions[0][-1] + precisions[1][-1] + precisions[2][-1]) / 3),
            'recall': tensor2val((recalls[0][-1] + recalls[1][-1] + recalls[2][-1]) / 3)
        }
        map_accum += average_results['map']
        writer.writerow(
            [t, 'car', car_results['map'], car_results['precision'], car_results['recall'], car_results['TP'],
             car_results['FP']])
        writer.writerow([t, 'person', person_results['map'], person_results['precision'], person_results['recall'],
                         person_results['TP'], person_results['FP']])
        writer.writerow(
            [t, 'rider', rider_results['map'], rider_results['precision'], rider_results['recall'], rider_results['TP'],
             rider_results['FP']])
        writer.writerow([t, 'average', average_results['map'], average_results['precision'], average_results['recall']])
    writer.writerow([t, 'mAP', map_accum / len(thresh)])
    fout.close()


# function to read predicted and annotated files
# returns list of ground truth results and predicted results
# function = 'classify'/'OD'
# results_fname = path to formatted results file
def read_inputs(func, results_fname):
    if func == 'classify':
        gt_select = 'rider'
    else:
        gt_select = 'bike_person'
    gt = []  # list of ground truth results
    results = []  # list of predicted results
    # read in ground truth annotations and add to list
    gt_file = open(gt_files[gt_select], 'r', newline='')
    gt_reader = csv.reader(gt_file, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
    for row in gt_reader:
        gt.append(row)
    # read in prediction results and add to list
    results_file = open(results_fname, 'r', newline='')
    results_reader = csv.reader(results_file, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
    for row in results_reader:
        results.append(row)
    return gt, results


if __name__ == "__main__":
    # example use: read in formatted classification results
    # calculates accuracy and iou to output files
    exp = 4
    gt, results = read_inputs('detect', './results/exp-{}/formatted.csv'.format(exp))
    coco_map(results, gt, 0.1, 0.95, 0.05, './results/exp-{}/map.csv'.format(exp))
    calculate_iou(results, gt, './results/exp-{}/iou.csv'.format(exp))
