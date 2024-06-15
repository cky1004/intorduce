import argparse
import preprocess_img
import cv2
import time
from ultralytics import YOLO 

parser = argparse.ArgumentParser(description="Argparse Tutorial")
parser.add_argument("--setting")
args = parser.parse_args()


def main(setting, img_path, model,_conf_num=0.3):
    start = time.time()
    testinpector = preprocess_img.FSinspector(img_path)
    print(img_path.split('/')[-1].split('.')[0])
    img_name = img_path.split('/')[-1].split('.')[0]
    final_img , count_data = testinpector.predict(model,conf_num=_conf_num)
    end = time.time()
    print(count_data)
    cv2.imwrite(f'./result/test_{img_name}.png', final_img)
    
    print(f"{end - start:.5f} sec")

if __name__=="__main__":
    # Model Load
    import torch
    import os
    model_path ='./inspect_code/yolo_model/best_total_231019.pt'
    model = YOLO(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    path_ = './test_img/'
    img_list = os.listdir(path_)
    img_paths = [os.path.join(path_,tmp) for tmp in img_list]

    print('시작')
    for img_path in img_paths:
        main(args.setting, img_path, model)
