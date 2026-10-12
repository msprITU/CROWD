from __future__ import division

import os
import warnings
from collections import OrderedDict
from config import return_args, args
from scipy.ndimage.filters import gaussian_filter
from torchvision import transforms
from utils import setup_seed
import nni
from nni.utils import merge_parameter
import util.misc as utils
import torch
import numpy as np
import cv2
import torch.nn as nn
from Networks.CDETR_Visdrone import build_model
import h5py #Bekir, Esad 2024
import time #Bekir, Esad 2024
import torch.nn.functional as F #Bekir, Esad 2024count

from pathlib import Path

## Line 61: Specify which set to run inference on — path to the .npy file.
## Line 82: Set based on the output folder and destination of the output .txt file;

img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tensor_transform = transforms.ToTensor()

warnings.filterwarnings('ignore')
'''fixed random seed '''
setup_seed(args.seed)


def main(args):

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(repo_root, 'data', 'visdrone_data')

    train_dataset = os.path.join(
        data_root, 'train', 'visdrone_train.npy'
    )
    test_dataset = os.path.join(
        data_root, 'test', 'visdrone_test.npy'
    )
        
    # Change the Dataset from here -Yagiz
    dataset = test_dataset
    print("Using: ", dataset)
    
    npy_file = np.load(dataset, allow_pickle=True) #With AUG Bekir, Esad 2024
    
    tests_dir = os.path.join(repo_root, 'tests')
    txt_file_path_inference = os.path.join(
        tests_dir, 'planner.txt'
    )

    txt_file_path_eval = os.path.join(
        tests_dir, 'eval_planner.txt'
    )
    with open(txt_file_path_eval, "w"):
        pass
    
    inference_models = []

    with open(txt_file_path_inference, 'r') as f:
        for line in f:
            print("Added model to queue: ", line.strip())
            inference_models.append(line.strip())

    if len(inference_models) != len(args['queries_list']): return "ERROR: queries_list and queue length mismatch."

    for i, model_path in enumerate(inference_models):
            
            num_queries = args['queries_list'][i]
            args['num_queries'] = num_queries
        
            utils.init_distributed_mode(return_args)
            model, criterion, postprocessors = build_model(return_args)
            model = model.cuda()

            model = nn.DataParallel(model, device_ids=[0])

            print("Processing the inference of model:", model_path)
            print("Using Queries:", num_queries)
            
                # checkpoint = torch.load(args['pre'])['state_dict'] # original code
            checkpoint = torch.load(model_path, map_location='cuda:0')['state_dict'] #Bekir, Esad 2024
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                # if 'backbone' in k or 'transformer' in k:
                    name = k.replace('bbox', 'point') # remove module.
                    new_state_dict[name] = v

            print("=> loading checkpoint '{}'".format(model_path))
                # checkpoint = torch.load(args['pre']) # original code
            model.load_state_dict(new_state_dict)
        
        

        # fourcc = cv2.VideoWriter_fourcc(*'XVID') # original code

        # cap = cv2.VideoCapture(args['video_path']) # original code

        # '''out video'''
        # width = 1024 # original code
        # height = 768 # original code
        # out = cv2.VideoWriter('./out_video.avi', fourcc, 30, (width*2, height*2))

        # while True: # original code
        #     try:    # original code
        #         ret, frame = cap.read() # original code
        #         frame = cv2.resize(frame, (width, height)) # original code
        #     except: # original code
        #         print("test end") # original code
        #         cap.release() # original code
        #         break # original code 
        

            ''' Process each h5 file in npy file ''' 
            model_file = Path(model_path).parent.name
            path = os.path.join(tests_dir, 'inference_outputs', f'inference_{model_file}.txt')
            with open(path, "w") as file: #Bekir, Esad 2024

                print("Writing to path:", path) 
                file.write(f"Processing the inference of model: {model_path}\n")
                file.write(f"Number of queries: {num_queries}\n")

                start_time = time.time() #Bekir, Esad 2024
                progress = 0 #Bekir, Esad 2024
                # counter = 0

                for h5_path in npy_file: #Bekir, Esad 2024
                    # if counter >= 5:  
                    #     break
                    
                    h5_path = h5_path.replace('.jpg', '.h5').replace('images_2048', 'gt_detr_map_2048') #Bekir, Esad 2024
                    print(h5_path) #Bekir, Esad 2024
                    file.write(h5_path + "\n") #Bekir, Esad 2024

                    with h5py.File(h5_path, 'r') as gt_file: #Bekir, Esad 2024
                        img = np.asarray(gt_file['image'])  # Load image from h5 file #Bekir, Esad 2024
                        
                        ''' Add padding to make dimensions multiples of 256 '''
                        padding_h = img.shape[0] % 256 #Bekir, Esad 2024 from dataset.py
                        padding_w = img.shape[1] % 256 #Bekir, Esad 2024 from dataset.py
                        # print(img.shape[1], img.shape[0]) #Bekir, Esad 2024 from dataset.py

                        if padding_h != 0: #Bekir, Esad 2024 from dataset.py
                            padding_h = 256 - padding_h #Bekir, Esad 2024 from dataset.py
                        if padding_w != 0: #Bekir, Esad 2024 from dataset.py
                            padding_w = 256 - padding_w #Bekir, Esad 2024 from dataset.py

                        pd = (padding_w, 0, padding_h, 0) #Bekir, Esad 2024 from dataset.py
                        img = F.pad(torch.from_numpy(img).permute(2, 0, 1), pd, 'constant', 0)  #Bekir, Esad 2024 from dataset.py
                        
                        # frame = frame.copy() # original code
                        frame = img.permute(1, 2, 0).numpy().copy() #Bekir, Esad 2024
                        ori_frame = frame.copy()
                        image = tensor_transform(frame)
                        image = img_transform(image)

                        width, height = image.shape[2], image.shape[1]
                        # print(width, height) # Bekir
                        num_w = int(width / 256)
                        num_h = int(height / 256)

                        image = image.view(3, num_h, 256, width).view(3, num_h, 256, num_w, 256)
                        image = image.permute(0, 1, 3, 2, 4).contiguous().view(3, num_w * num_h, 256, 256).permute(1, 0, 2, 3)


                        with torch.no_grad():
                            image = image.cuda()
                            outputs = model(image)

                            out_logits, out_point = outputs['pred_logits'], outputs['pred_points']

                            prob = out_logits.sigmoid()
                            probmax, label_ind = torch.max(prob, dim=2)
                            
                            # for j in range(6):
                            #   for i in range(700):
                            #       if label_ind[j, i] > 1 and  prob[j, i, label_ind[j, i]] > 0.009:
                            #         print("FOUND NON CAR :", label_ind[j, i], prob[j, i])

                            # prob2, label_ind = torch.max(prob, dim=2)
        
                            topk_values, topk_points = torch.topk(probmax, args['num_queries'], dim=1)

                            out_point = torch.gather(out_point, 1, topk_points.unsqueeze(-1).repeat(1, 1, 2))
                            out_point = out_point * 256
                            
                            out_labels = torch.gather(label_ind, 1, topk_points)

                            value_points = torch.cat([topk_values.unsqueeze(2), out_point, out_labels.unsqueeze(2)], 2)
                            crop_size = 256

                            # kpoint_map, density_map, frame, count = show_map(value_points, frame, width, height, crop_size, num_h, num_w) # original code


                            # res1 = np.hstack((ori_frame, kpoint_map)) # original code
                            # res2 = np.hstack((density_map, frame))    # original code
                            # res = np.vstack((res1, res2))             # original code

                            # cv2.putText(res, "Count:" + str(count), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5) # original code

                            # out.write(res) # original code
                            # print('count:', count) # original code


                            pred_coor, cls_label, cls_prob = show_map(value_points, frame, width, height, crop_size, num_h, num_w) #Bekir, Esad 2024
                            
                            for j in range(len(pred_coor[0])): #Bekir, Esad 2024
                                # corrected_x = int(pred_coor[1][j]) - (padding_w // 2) #Bekir, Esad 2024
                                # corrected_y = int(pred_coor[0][j]) - (padding_h // 2) #Bekir, Esad 2024
                                corrected_x = int(pred_coor[1][j]) - (padding_w) #Bekir, Esad 2024
                                corrected_y = int(pred_coor[0][j]) - (padding_h) #Bekir, Esad 2024
                                file.write(str(corrected_x) + ', ' + str(corrected_y) + ', ' + str(int(cls_label[j])) + ', ' + str(cls_prob[j]) + '\n') #Bekir, Esad 2024
                            
                            file.write("\n")  # Separate different images with a new line #Bekir, Esad 2024

                    # counter += 1
                    progress += 1 #Bekir, Esad 2024
                    if progress % 100 == 0: #Bekir, Esad 2024
                        print(progress, round((time.time() - start_time), 2)) #Bekir, Esad 2024
                        start_time = time.time() #Bekir, Esad 2024

            file.close() #Bekir, Esad 2024

            with open(txt_file_path_eval, "a") as file:
                file.write(f'{path}\n')

def show_map(out_pointes, frame, width, height, crop_size, num_h, num_w):
    kpoint_list = []
    confidence_list = []

    for i in range(len(out_pointes)):
      
        out_value = out_pointes[i].squeeze(0)[:, 0].data.cpu().numpy()
        out_point = out_pointes[i].squeeze(0)[:, 1:3].data.cpu().numpy().tolist()
        out_class = out_pointes[i].squeeze(0)[:, 3].data.cpu().numpy()

        k = np.zeros((crop_size, crop_size))
        c_map = np.zeros((crop_size, crop_size))

        '''get coordinate'''
        for j in range(len(out_point)):
            
            if out_class[j] <= 1.1:
                if out_value[j] >= 0.35:
                    x = int(out_point[j][0])
                    y = int(out_point[j][1])
                    k[x, y] = out_class[j]
                    c_map[x, y] = out_value[j]
            else:
                if out_value[j] >= 0.35:
                    x = int(out_point[j][0])
                    y = int(out_point[j][1])  
                    k[x, y] = out_class[j]
                    c_map[x, y] = out_value[j]

            if out_value[j] < 0.35:
                break

        kpoint_list.append(k)
        confidence_list.append(c_map)

    kpoint = torch.from_numpy(np.array(kpoint_list)).unsqueeze(0)
    kpoint = kpoint.view(num_h, num_w, crop_size, crop_size).permute(0, 2, 1, 3).contiguous().view(num_h, crop_size,
                                                                                                   width).view(height,
                                                                                                               width).cpu().numpy()

    c_map = torch.from_numpy(np.array(confidence_list)).unsqueeze(0)
    c_map = c_map.view(num_h, num_w, crop_size, crop_size).permute(0, 2, 1, 3).contiguous().view(num_h, crop_size,
                                                                                                   width).view(height,
                                                                                                               width).cpu().numpy()

    '''obtain the coordinate '''
    pred_coor = np.nonzero(kpoint)
    count = len(pred_coor[0])


    # return point_map, density_map, frame, count
    return pred_coor, kpoint[pred_coor], c_map[pred_coor] #Bekir, Esad 2024


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()

    params = vars(merge_parameter(return_args, tuner_params))

    main(params)