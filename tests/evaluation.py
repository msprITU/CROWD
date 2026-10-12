import h5py
import os
import numpy as np
import cv2
import math
import json
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from pathlib import Path

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tests_dir = os.path.dirname(os.path.abspath(__file__))  # Tests directory
eval_planner_path = os.path.join(tests_dir, 'eval_planner.txt')  # Evaluation planner
eval_results_dir = os.path.join(tests_dir, 'eval_results')  # Evaluation results directory
os.makedirs(eval_results_dir, exist_ok=True)

print()
print("SELECTIVE METHOD IS USED - MAKE SURE RIGHT CLASSES ARE SELECTED")
print()

sigma_value = 8

## Hangi sınıfların işleneceği 196.-199. satırlar arası belirlenmel
## k[k == X] = X -> X sınıfı işlenir  |||   k[k == X] = 0 -> X sınıfı işlenmez 

## 155. Satır inference çıktısının pathine göre düzenlenmeli
## 156. Satır test sonucunun kaydedileceği pathe göre düzenlenmeli


# Cost matrix calculation with Euclidean distance
def compute_euclidean_cost(gt_points, proposal_points):
    # Euclidean distance calculation
    euclidean_cost_matrix = distance.cdist(proposal_points, gt_points, 'euclidean')
    return euclidean_cost_matrix

# Greedy matching method
def greedy_matching(euclidean_cost_matrix, sigma_values):
    matched_pairs = []
    used_proposals = set()
    used_gt_points = set()

    for proposal_idx, distances in enumerate(euclidean_cost_matrix):
        if proposal_idx in used_proposals:
            continue  # Daha önce eşleştirildiyse atla

        if distances.shape[0] == 0:
            continue
        
        closest_gt_idx = np.argmin(distances)
        closest_distance = distances[closest_gt_idx]

        if closest_distance <= sigma_values and closest_gt_idx not in used_gt_points:
            matched_pairs.append((proposal_idx, closest_gt_idx))
            used_proposals.add(proposal_idx)
            used_gt_points.add(closest_gt_idx)

    return matched_pairs

# Hungarian matching based on Euclidean cost matrix (commented out)
# def euclidean_hungarian_matching(euclidean_cost_matrix, sigma_values):
#     row_ind, col_ind = linear_sum_assignment(euclidean_cost_matrix)
# 
#     matched_pairs = []
#     for row, col in zip(row_ind, col_ind):
#         if euclidean_cost_matrix[row, col] <= sigma_values:  # comparison with sigma
#             matched_pairs.append((row, col))  # Matching (proposal, GT) pairs
# 
#     return matched_pairs

# TP, FP, FN calculations and counting metrics (MAE, MSE) with Euclidean distance
def compute_metrics_euclidean_with_sigma(loc_list, gt_list, sigma_values):
    # Create Euclidean based cost matrix

    total_pred = []
    for loc_array in loc_list:
        total_pred.append(loc_array.shape[0])
        # print(total_pred)

    total_gt = []
    for gt_array in gt_list:
        total_gt.append(gt_array.shape[0])
        # print(total_gt)

    tp_list_list = []
    tp_list_list.append([])
    tp_list_list.append([])
    tp_list_list.append([])
    tp_list_list.append([])
    tp_list_list.append([])
    tp_list_list.append([])
    tp_list_list.append([])
    tp_list_list.append([])

    matched_pairs_list = []

    for loc_array, gt_array, tp_list in zip(loc_list, gt_list, tp_list_list):
        euclidean_cost_matrix = compute_euclidean_cost(gt_array, loc_array)
        # Use Greedy matching instead of Hungarian
        matched_pairs = greedy_matching(euclidean_cost_matrix, sigma_values)
        for pair in matched_pairs:
            tp_list.append(pair)
        matched_pairs_list.append(matched_pairs)
    

    # # TP, FP, FN calculations
    # matched_proposal_indices = {pair[0] for pair in matched_pairs}

    tp1 = len(tp_list_list[0])
    fp1 = total_pred[0] - len(tp_list_list[0])
    fn1 = total_gt[0] - len(tp_list_list[0])

    tp2 = len(tp_list_list[1])
    fp2 = total_pred[1] - len(tp_list_list[1])
    fn2 = total_gt[1] - len(tp_list_list[1])

    tp3 = len(tp_list_list[2])
    fp3 = total_pred[2] - len(tp_list_list[2])
    fn3 = total_gt[2] - len(tp_list_list[2])

    tp4 = len(tp_list_list[3])
    fp4 = total_pred[3] - len(tp_list_list[3])
    fn4 = total_gt[3] - len(tp_list_list[3])

    tp5 = len(tp_list_list[4])
    fp5 = total_pred[4] - len(tp_list_list[4])
    fn5 = total_gt[4] - len(tp_list_list[4])

    tp6 = len(tp_list_list[5])
    fp6 = total_pred[5] - len(tp_list_list[5])
    fn6 = total_gt[5] - len(tp_list_list[5])

    tp7 = len(tp_list_list[6])
    fp7 = total_pred[6] - len(tp_list_list[6])
    fn7 = total_gt[6] - len(tp_list_list[6])

    tp8 = len(tp_list_list[7])
    fp8 = total_pred[7] - len(tp_list_list[7])
    fn8 = total_gt[7] - len(tp_list_list[7])

    logfile.write("-"*15)
    logfile.write(f"\nTest Count: {test_count}\n")
    logfile.write(f"Class 1: TP: {tp1} FP: {fp1} FN: {fn1} GT COUNT: {len(gt_list[0])}\n")
    logfile.write(f"Class 2: TP: {tp2} FP: {fp2} FN: {fn2} GT COUNT: {len(gt_list[1])}\n")
    logfile.write(f"Class 3: TP: {tp3} FP: {fp3} FN: {fn3} GT COUNT: {len(gt_list[2])}\n")
    logfile.write(f"Class 4: TP: {tp4} FP: {fp4} FN: {fn4} GT COUNT: {len(gt_list[3])}\n")
    logfile.write(f"Class 5: TP: {tp5} FP: {fp5} FN: {fn5} GT COUNT: {len(gt_list[4])}\n")
    logfile.write(f"Class 6: TP: {tp6} FP: {fp6} FN: {fn6} GT COUNT: {len(gt_list[5])}\n")
    logfile.write(f"Class 7: TP: {tp7} FP: {fp7} FN: {fn7} GT COUNT: {len(gt_list[6])}\n")
    logfile.write(f"Class 8: TP: {tp8} FP: {fp8} FN: {fn8} GT COUNT: {len(gt_list[7])}\n")
    
    class_counts = [len(gt_list[0]), len(gt_list[1]), len(gt_list[2]), len(gt_list[3]), len(gt_list[4]), len(gt_list[5]), len(gt_list[6]), len(gt_list[7])]

    # Counting Metrics (MAE, MSE)
    abs_error = []
    squared_error = []

    Pi = total_pred[0]
    Gi = total_gt[0]
    error = abs(Pi - Gi)
    abs_error.append(error)
    squared_error.append(error ** 2)

    Pi = total_pred[1]
    Gi = total_gt[1]
    error = abs(Pi - Gi)
    abs_error.append(error)
    squared_error.append(error ** 2)

    Pi = total_pred[2]
    Gi = total_gt[2]
    error = abs(Pi - Gi)
    abs_error.append(error)
    squared_error.append(error ** 2)

    Pi = total_pred[3]
    Gi = total_gt[3]
    error = abs(Pi - Gi)
    abs_error.append(error)
    squared_error.append(error ** 2)

    Pi = total_pred[4]
    Gi = total_gt[4]
    error = abs(Pi - Gi)
    abs_error.append(error)
    squared_error.append(error ** 2)

    Pi = total_pred[5]
    Gi = total_gt[5]
    error = abs(Pi - Gi)
    abs_error.append(error)
    squared_error.append(error ** 2)

    Pi = total_pred[6]
    Gi = total_gt[6]
    error = abs(Pi - Gi)
    abs_error.append(error)
    squared_error.append(error ** 2)

    Pi = total_pred[7]
    Gi = total_gt[7]
    error = abs(Pi - Gi)
    abs_error.append(error)
    squared_error.append(error ** 2)

    tp = []
    tp.append(tp1)
    tp.append(tp2)
    tp.append(tp3)
    tp.append(tp4)
    tp.append(tp5)
    tp.append(tp6)
    tp.append(tp7)
    tp.append(tp8)

    fp = []
    fp.append(fp1)
    fp.append(fp2)
    fp.append(fp3)
    fp.append(fp4)
    fp.append(fp5)
    fp.append(fp6)
    fp.append(fp7)
    fp.append(fp8)


    fn = []
    fn.append(fn1)
    fn.append(fn2)
    fn.append(fn3)
    fn.append(fn4)
    fn.append(fn5)
    fn.append(fn6)
    fn.append(fn7)
    fn.append(fn8)
    

    return tp, fp, fn, matched_pairs_list, abs_error, squared_error, class_counts

file_queue = []

with open(eval_planner_path, 'r') as f:
    for line in f:
        stripped_line = line.strip()
        if stripped_line:
            file_queue.append(stripped_line)

print(f"Inference files to be processed: (By order) {file_queue}")

for i, file_path in enumerate(file_queue):

    # Resolve relative inference paths against the tests directory.
    if not os.path.isabs(file_path):
        file_path = os.path.join(tests_dir, file_path)

    model_file = Path(file_path).stem.replace("inference_", "")  #Bekir, Esad 2024
    logfile_path = os.path.join(
    eval_results_dir, f'result_{model_file}.txt'
) 
    print(f"Writing results to: {logfile_path}")

    if os.path.exists(logfile_path):
        logfile_path = logfile_path.replace('.txt', f'_{i}.txt')

    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Skipping.")

        with open(logfile_path, 'a') as f:
            f.write(file_path + " - File not found and skipped\n")
        continue

    folder_path = os.path.dirname(logfile_path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Output folder created at {os.path.basename(folder_path)}")

    test_count = 0
    total_abs_error = 0
    total_squared_error = 0
    prcnt = 0

    log_list = []
    precision_list_1 = []
    precision_list_2 = []
    precision_list_3 = []
    precision_list_4 = []
    precision_list_5 = []
    precision_list_6 = []
    precision_list_7 = []
    precision_list_8 = []
    recall_list_1 = []
    recall_list_2 = []
    recall_list_3 = []
    recall_list_4 = []
    recall_list_5 = []
    recall_list_6 = []
    recall_list_7 = []
    recall_list_8 = []
    fone_list_1 = []
    fone_list_2 = []
    fone_list_3 = []
    fone_list_4 = []
    fone_list_5 = []
    fone_list_6 = []
    fone_list_7 = []
    fone_list_8 = []


    total_gt_class1 = 0
    total_gt_class2 = 0
    total_gt_class3 = 0
    total_gt_class4 = 0
    total_gt_class5 = 0
    total_gt_class6 = 0
    total_gt_class7 = 0
    total_gt_class8 = 0

    with open(file_path, 'r') as file, open(logfile_path, 'w') as logfile:
        logfile.write("Evaluation Results for: " + file_path + '\n')
        logfile.write("Inference file: " + file_path + '\n')
        for line in file:
            stripped_line = line.strip()
            
            if stripped_line.startswith("/"):
                
                proposal_list = []
                gt_path = stripped_line
                test_count += 1
                gt_list_list = []

                c_counts = np.zeros(8)

                while True:
                    try:
                        gt_file = h5py.File(gt_path, 'r')                    
                        k = np.asarray(gt_file['kpoint'])

                        ## WHICH CLASSES WILL BE PROCESSED? - ESAD
                        k[k == 1] = 1
                        k[k == 2] = 2
                        k[k == 3] = 3
                        k[k == 4] = 4
                        k[k == 5] = 5
                        k[k == 6] = 6
                        k[k == 7] = 7
                        k[k == 8] = 8

                        gt_list = np.array(np.nonzero(k)).T
                        labels = k[np.nonzero(k)]
                        gt_list_list.append(gt_list[labels == 1])
                        gt_list_list.append(gt_list[labels == 2])
                        gt_list_list.append(gt_list[labels == 3])
                        gt_list_list.append(gt_list[labels == 4])
                        gt_list_list.append(gt_list[labels == 5])
                        gt_list_list.append(gt_list[labels == 6])
                        gt_list_list.append(gt_list[labels == 7])
                        gt_list_list.append(gt_list[labels == 8])

                        total_gt_class1 += len(gt_list[labels == 1])
                        total_gt_class2 += len(gt_list[labels == 2])
                        total_gt_class3 += len(gt_list[labels == 3])
                        total_gt_class4 += len(gt_list[labels == 4])
                        total_gt_class5 += len(gt_list[labels == 5])
                        total_gt_class6 += len(gt_list[labels == 6])
                        total_gt_class7 += len(gt_list[labels == 7])
                        total_gt_class8 += len(gt_list[labels == 8])

                        break
                    except OSError:
                        cv2.waitKey(1000)


            elif not stripped_line:

                print(c_counts)
                prcnt += 1
                loc_array_list = []

                if prcnt >= 0 and gt_list.size != 0 and len(proposal_list) != 0:
                    prcnt = 0
                    proposal_array = np.array(proposal_list)
                    loc_array = proposal_array[:, :2]
                    clc_array = proposal_array[:, 2]
                    loc_array_list.append(loc_array[clc_array == 1])
                    loc_array_list.append(loc_array[clc_array == 2])
                    loc_array_list.append(loc_array[clc_array == 3])
                    loc_array_list.append(loc_array[clc_array == 4])
                    loc_array_list.append(loc_array[clc_array == 5])
                    loc_array_list.append(loc_array[clc_array == 6])
                    loc_array_list.append(loc_array[clc_array == 7])
                    loc_array_list.append(loc_array[clc_array == 8])

                    gt_count = gt_list.shape[0]
                    proposal_count = proposal_array.shape[0]

                    sigma_values = sigma_value  # for sha constant value

                    # Calculate tp, fp, fn, matched_pairs, abs_error, squared_error with Euclidean-based Greedy matching and sigma values
                    tp, fp, fn, matched_pairs_list, abs_error, squared_error, cnts = compute_metrics_euclidean_with_sigma(loc_array_list, gt_list_list, sigma_values)

                    gt_count = np.count_nonzero(k)
                    matched_count = sum(len(sublist) for sublist in matched_pairs_list)
    
                    total_abs_error += sum(abs_error)
                    total_squared_error += sum(squared_error)

                    log_tpl = (gt_count, matched_count, len(proposal_list))
                    log_list.append(log_tpl)
                    
                    logfile.write(gt_path + '\n')
                    logfile.write(str(log_tpl) + '\n')

                    tot_dist = 0
                    if matched_count > 0:
                        for matched_pairs in matched_pairs_list:
                            for pair in matched_pairs:
                                euclidean_distance = math.sqrt((gt_list[pair[1]][0] - loc_array[pair[0]][0]) ** 2 +
                                                            (gt_list[pair[1]][1] - loc_array[pair[0]][1]) ** 2)
                                tot_dist += euclidean_distance

                                formatted_pair = "\t[{0:>4}, {1:>4}], [{2:>4}, {3:>4}]  Euclidean distance: {4}".format(
                                                    gt_list[pair[1]][0], gt_list[pair[1]][1], loc_array[pair[0]][0], loc_array[pair[0]][1], euclidean_distance)
                                logfile.write(formatted_pair + '\n')

                        avg_dist = tot_dist / matched_count
                        logfile.write('\n' + 'Total distance: ' + str(tot_dist) + '\tAverage distance: ' + str(avg_dist) + '\n')
                    else:
                        logfile.write('\nNo matched pairs found. Total distance: ' + str(tot_dist) + '\tAverage distance: N/A (no matches)\n')

                    epsilon = 1e-10

                    # Yagiz: Logic here is simple, If the False positives and Ground truths are zero, then it's basically nonsense for us to 
                    # Calculate precision, recall and f1 for this class. This optimization should make the code faster and more accurate
                    one,two,three,four,five,six,seven,eight = False, False, False, False, False, False, False, False

                    if fp[0] != 0 or cnts[0] != 0: 
                      precision1 = tp[0] / (tp[0] + fp[0] + epsilon)
                      recall1 = tp[0] / (tp[0] + fn[0] + epsilon)
                      fone1 = 2 * tp[0] / (2 * tp[0] + fp[0] + fn[0] + epsilon)

                      precision_list_1.append(precision1)
                      recall_list_1.append(recall1)
                      fone_list_1.append(fone1)

                      one = True

                    
                    if fp[1] != 0 or cnts[1] != 0: 
                      precision2 = tp[1] / (tp[1] + fp[1] + epsilon)
                      recall2 = tp[1] / (tp[1] + fn[1] + epsilon)
                      fone2 = 2 * tp[1] / (2 * tp[1] + fp[1] + fn[1] + epsilon)

                      precision_list_2.append(precision2)
                      recall_list_2.append(recall2)
                      fone_list_2.append(fone2)

                      two = True

                    if fp[2] != 0 or cnts[2] != 0: 
                      precision3 = tp[2] / (tp[2] + fp[2] + epsilon)
                      recall3 = tp[2] / (tp[2] + fn[2] + epsilon)
                      fone3 = 2 * tp[2] / (2 * tp[2] + fp[2] + fn[2] + epsilon)

                      precision_list_3.append(precision3)
                      recall_list_3.append(recall3)
                      fone_list_3.append(fone3)

                      three = True

                    if fp[3] != 0 or cnts[3] != 0: 
                      precision4 = tp[3] / (tp[3] + fp[3] + epsilon)
                      recall4 = tp[3] / (tp[3] + fn[3] + epsilon)
                      fone4 = 2 * tp[3] / (2 * tp[3] + fp[3] + fn[3] + epsilon)

                      precision_list_4.append(precision4)
                      recall_list_4.append(recall4)
                      fone_list_4.append(fone4)

                      four = True

                    if fp[4] != 0 or cnts[4] != 0: 
                      precision5 = tp[4] / (tp[4] + fp[4] + epsilon)
                      recall5 = tp[4] / (tp[4] + fn[4] + epsilon)
                      fone5 = 2 * tp[4] / (2 * tp[4] + fp[4] + fn[4] + epsilon)

                      precision_list_5.append(precision5)
                      recall_list_5.append(recall5)
                      fone_list_5.append(fone5)

                      five = True

                    if fp[5] != 0 or cnts[5] != 0: 
                      precision6 = tp[5] / (tp[5] + fp[5] + epsilon)
                      recall6 = tp[5] / (tp[5] + fn[5] + epsilon)
                      fone6 = 2 * tp[5] / (2 * tp[5] + fp[5] + fn[5] + epsilon)

                      precision_list_6.append(precision6)
                      recall_list_6.append(recall6)
                      fone_list_6.append(fone6)

                      six = True

                    if fp[6] != 0 or cnts[6] != 0: 
                      precision7 = tp[6] / (tp[6] + fp[6] + epsilon)
                      recall7 = tp[6] / (tp[6] + fn[6] + epsilon)
                      fone7 = 2 * tp[6] / (2 * tp[6] + fp[6] + fn[6] + epsilon)

                      precision_list_7.append(precision7)
                      recall_list_7.append(recall7)
                      fone_list_7.append(fone7)

                      seven = True

                    if fp[7] != 0 or cnts[7] != 0: 
                      precision8 = tp[7] / (tp[7] + fp[7] + epsilon)
                      recall8 = tp[7] / (tp[7] + fn[7] + epsilon)
                      fone8 = 2 * tp[7] / (2 * tp[7] + fp[7] + fn[7] + epsilon)

                      precision_list_8.append(precision8)
                      recall_list_8.append(recall8)
                      fone_list_8.append(fone8)

                      eight = True


                    #yoruma alınabilir
                    #if matched_count > 0:
                        #print(test_count)
                        #print(log_tpl)
                    if one: logfile.write(f'Precision1: {round(precision1, 3)}   Recall1: {round(recall1, 3)}   F1: {round(fone1, 3)}\n')
                    if two: logfile.write(f'Precision2: {round(precision2, 3)}   Recall2: {round(recall2, 3)}   F1: {round(fone2, 3)}\n')
                    if three: logfile.write(f'Precision3: {round(precision3, 3)}   Recall3: {round(recall3, 3)}   F1: {round(fone3, 3)}\n')
                    if four: logfile.write(f'Precision4: {round(precision4, 3)}   Recall4: {round(recall4, 3)}   F1: {round(fone4, 3)}\n')
                    if five: logfile.write(f'Precision5: {round(precision5, 3)}   Recall5: {round(recall5, 3)}   F1: {round(fone5, 3)}\n')
                    if six: logfile.write(f'Precision6: {round(precision6, 3)}   Recall6: {round(recall6, 3)}   F1: {round(fone6, 3)}\n')
                    if seven: logfile.write(f'Precision7: {round(precision7, 3)}   Recall7: {round(recall7, 3)}   F1: {round(fone7, 3)}\n')
                    if eight: logfile.write(f'Precision8: {round(precision8, 3)}   Recall8: {round(recall8, 3)}   F1: {round(fone8, 3)}\n')
                        

                        # img = np.asarray(gt_file['image'])
                        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # for i in gt_list:
                        #     cv2.circle(img, (int(i[1]), int(i[0])), 3, (255, 0, 0), -1)
                        # for i in proposal_list:
                        #     cv2.circle(img, (int(i[1]), int(i[0])), 4, (0, 255, 255), -1)
                        # for pair in matched_pairs:
                        #     point1 = (int(gt_list[pair[1]][1]), int(gt_list[pair[1]][0]))
                        #     point2 = (int(proposal_array[pair[0]][1]), int(proposal_array[pair[0]][0]))
                        #     cv2.line(img, point1, point2, (0, 0, 255), 1)

                        # cv2.putText(img, str(gt_count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        # cv2.putText(img, str(len(proposal_list)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        # cv2.putText(img, str(matched_count), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        # cv2.putText(img, f'{round(precision1, 3)}   {round(recall1, 3)}   {round(fone1, 3)}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        # cv2.putText(img, f'{round(precision2, 3)}   {round(recall2, 3)}   {round(fone2, 3)}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        # cv2.putText(img, f'{round(precision3, 3)}   {round(recall3, 3)}   {round(fone3, 3)}', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                        # try:
                        #     cv2.imwrite('/content/drive/MyDrive/VTransformer/CLTR-main/Test_Files/deneme123/' + str(test_count) + '.png', img)
                        # except Exception as e:
                        #     print(f"Error saving image: {e}")
                        #     continue
                    # yoruma al


            else:
                if ',' not in stripped_line:
                    print(f"Skipping line without comma (Expected within first lines): {stripped_line}")
                    continue

                numbers = stripped_line.split(',')
                num1, num2, num3 = map(int, numbers[:3])
                item = (num2, num1, num3)
                
                c_counts[num3-1] += 1

                proposal_list.append(item)

    # Calculate final MAE and MSE
        mae = total_abs_error / test_count
        mse = math.sqrt(total_squared_error / test_count)
        print('MAE: ', mae, 'MSE: ', mse)

        precision_1 = sum(precision_list_1) / (len(precision_list_1) + epsilon)
        recall_1 = sum(recall_list_1) / (len(recall_list_1) + epsilon)
        fone_1 = sum(fone_list_1) / (len(fone_list_1) + epsilon)

        precision_2 = sum(precision_list_2) / (len(precision_list_2) + epsilon)
        recall_2 = sum(recall_list_2) / (len(recall_list_2) + epsilon)
        fone_2 = sum(fone_list_2) / (len(fone_list_2) + epsilon)

        precision_3 = sum(precision_list_3) / (len(precision_list_3) + epsilon)
        recall_3 = sum(recall_list_3) / (len(recall_list_3) + epsilon)
        fone_3 = sum(fone_list_3) / (len(fone_list_3) + epsilon)

        precision_4 = sum(precision_list_4) / (len(precision_list_4) + epsilon)
        recall_4 = sum(recall_list_4) / (len(recall_list_4) + epsilon)
        fone_4 = sum(fone_list_4) / (len(fone_list_4) + epsilon)

        precision_5 = sum(precision_list_5) / (len(precision_list_5) + epsilon)
        recall_5 = sum(recall_list_5) / (len(recall_list_5) + epsilon)
        fone_5 = sum(fone_list_5) / (len(fone_list_5) + epsilon)

        precision_6 = sum(precision_list_6) / (len(precision_list_6) + epsilon)
        recall_6 = sum(recall_list_6) / (len(recall_list_6) + epsilon)
        fone_6 = sum(fone_list_6) / (len(fone_list_6) + epsilon)

        precision_7 = sum(precision_list_7) / (len(precision_list_7) + epsilon)
        recall_7 = sum(recall_list_7) / (len(recall_list_7) + epsilon)
        fone_7 = sum(fone_list_7) / (len(fone_list_7) + epsilon)

        precision_8 = sum(precision_list_8) / (len(precision_list_8) + epsilon)
        recall_8 = sum(recall_list_8) / (len(recall_list_8) + epsilon)
        fone_8 = sum(fone_list_8) / (len(fone_list_8) + epsilon)


        logfile.write('\nFinal Metrics:\n')
        logfile.write(f'MAE: {mae}, MSE: {mse}\n')
        logfile.write(f'Precision Class 1: {precision_1}, Recall Class 1: {recall_1}, F1 Class 1: {fone_1}\n')
        logfile.write(f'Precision Class 2: {precision_2}, Recall Class 2: {recall_2}, F1 Class 2: {fone_2}\n')
        logfile.write(f'Precision Class 3: {precision_3}, Recall Class 3: {recall_3}, F1 Class 3: {fone_3}\n')
        logfile.write(f'Precision Class 4: {precision_4}, Recall Class 4: {recall_4}, F1 Class 4: {fone_4}\n')
        logfile.write(f'Precision Class 5: {precision_5}, Recall Class 5: {recall_5}, F1 Class 5: {fone_5}\n')
        logfile.write(f'Precision Class 6: {precision_6}, Recall Class 6: {recall_6}, F1 Class 6: {fone_6}\n')
        logfile.write(f'Precision Class 7: {precision_7}, Recall Class 7: {recall_7}, F1 Class 7: {fone_7}\n')
        logfile.write(f'Precision Class 8: {precision_8}, Recall Class 8: {recall_8}, F1 Class 8: {fone_8}\n')
        logfile.write(f'Total GT Class 1: {total_gt_class1}, Total GT Class 2: {total_gt_class2}, Total GT Class 3: {total_gt_class3}, Total GT Class 4: {total_gt_class4}, Total GT Class 5: {total_gt_class5}, Total GT Class 6: {total_gt_class6}, Total GT Class 7: {total_gt_class7}, Total GT Class 8: {total_gt_class8}\n')

        print()
        print("Class 1")
        print("Precision:", precision_1)
        print("Recall:", recall_1)
        print("F1:", fone_1)

        print("Class 2")
        print("Precision:", precision_2)
        print("Recall:", recall_2)
        print("F1:", fone_2)

        print("Class 3")
        print("Precision:", precision_3)
        print("Recall:", recall_3)
        print("F1:", fone_3)

        print("Class 4")
        print("Precision:", precision_4)
        print("Recall:", recall_4)
        print("F1:", fone_4)

        print("Class 5")
        print("Precision:", precision_5)
        print("Recall:", recall_5)
        print("F1:", fone_5)

        print("Class 6")
        print("Precision:", precision_6)
        print("Recall:", recall_6)
        print("F1:", fone_6)

        print("Class 7")
        print("Precision:", precision_7)
        print("Recall:", recall_7)
        print("F1:", fone_7)

        print("Class 8")
        print("Precision:", precision_8)
        print("Recall:", recall_8)
        print("F1:", fone_8)


    print("Sınıf 1: ", total_gt_class1)
    print("Sınıf 2: ", total_gt_class2)
    print("Sınıf 3: ", total_gt_class3)
    print("Sınıf 4: ", total_gt_class4)
    print("Sınıf 5: ", total_gt_class5)
    print("Sınıf 6: ", total_gt_class6)
    print("Sınıf 7: ", total_gt_class7)
    print("Sınıf 8: ", total_gt_class8)
