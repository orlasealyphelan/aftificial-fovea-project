import csv
from tools import format_json


# functions to format annotations and predictions to formatted csv file
# row = [frame, class id, confidence, xmin, ymin, xmax, ymax]
def ground_truth(out_writer, rider=True):
    if rider:
        json_file = 'annotations/rider/instances_default.json'
    else:
        json_file = 'annotations/bike_person/instances_default.json'
    _, results = format_json(json_file)
    for key in results:
        for res in results[key]:
            out_writer.writerow(
                [key - 1, res[0] - 1, 0, round(res[1], 2), round(res[2], 2),
                 round(res[3], 2), round(res[4], 2)])


def predictions(results, out, c=True):
    with open(results) as file:
        count = 0
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            if count == 0:
                count += 1  # skip first row of headings
            else:
                if c:
                    # classifier class ids same format as annotations
                    out.writerow(
                        [int(row[8]), int(row[6]), round(float(row[5]), 2), round(float(row[1]), 2),
                         round(float(row[2]), 2),
                         round(float(row[3]), 2), round(float(row[4]), 2)])
                else:
                    # YOLO: car: 2, person: 0, bicycle: 1
                    # change to same format as annotations:
                    # car: 0, person: 1, bicycle: 2
                    classes = {'2': 0, '0': 1, '1': 2}
                    if int(row[6]) < 3:
                        out.writerow(
                            [int(row[8]), int(classes[row[6]]), round(float(row[5]), 2), round(float(row[1]), 2),
                             round(float(row[2]), 2),
                             round(float(row[3]), 2), round(float(row[4]), 2)])


def run(pred, c, exp):
    if pred:
        if c:
            t = "classify"
        else:
            t = "detect"
        results_file = './results/exp-{}/{}.csv'.format(exp, t)
        output_file = open('./results/exp-{}/formatted.csv'.format(exp), 'w', newline='')
        output_writer = csv.writer(output_file)
        predictions(results_file, output_writer, c=c)
    else:
        if c:
            fout = open('annotations/rider/formatted_gt.csv', 'w', newline='')  # annotations for classifier
        else:
            fout = open('annotations/bike_person/formatted_gt.csv', 'w', newline='')  # annotations for YOLO
        writer = csv.writer(fout)
        ground_truth(writer, c)


if __name__ == "__main__":
    prediction = True  # predicted results - True, ground truth results - False
    classify = True  # classification results - True, YOLO results - False
    exp_num = 2  # experiment number to reformat
    run(prediction, classify, exp_num)
