import numpy as np
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO 
from collections import Counter
import multiprocessing

color_map ={
    0: (255, 255, 0),
    1: (0, 255, 255),
    2: (255, 0, 255),
}

class FSinspector:
    def __init__(self,img_path):
        
        self.path = img_path
        self.image = self.rotate_image_if_needed()
        self.row = 3
        self.col = 3
        self.size = 640
        self.x, self.y, self.w, self.h = 100, 1100, self.size*self.row, self.size*self.col
        self.roi = self.image[self.y:self.y+self.h, self.x:self.x+self.w]

    def process_sub_image(self, args):
        sub_img, model, conf_num = args
        results = model.predict(sub_img, conf=conf_num, verbose=False)
        t_cnt = []
        for box in results[0].boxes.cpu().numpy():
            r = box.xyxy[0].astype(int)
            cls = int(box.cls[0])
            proba = format(box.data[0][4],".1f")
            cv2.rectangle(sub_img, r[:2], r[2:], color_map[cls], 2)
            cv2.putText(sub_img, str(model.names[cls])+(f'{proba}%'), (r[:2][0]-10,r[:2][1]-10), cv2.FONT_ITALIC, 1, color_map[cls], 2)
            t_cnt.append(cls)
        return sub_img, t_cnt

    def predict(self, model, conf_num=0.25):
        height, width, channels = self.roi.shape
        sub_height = height // self.row
        sub_width = width // self.col
        sub_images = []
        for row in range(self.row):
            for col in range(self.col):
                sub_img = self.roi.copy()[row * sub_height: (row + 1) * sub_height,
                                col * sub_width: (col + 1) * sub_width, :]
                sub_images.append(sub_img)
          
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results = pool.map(self.process_sub_image, [(sub_img, model, conf_num) for sub_img in sub_images])
        pool.close()
        pool.join()

        return_data = []
        for result in results:
            sub_img, t_cnt = result
            return_data.append(t_cnt)
            
        class_count = Counter(model.names[class_] for sublist in return_data for class_ in sublist)
        final_img = self.combine_images([result[0] for result in results])
        return final_img, class_count

    def combine_images(self, sub_images):
        if len(sub_images) > 0 and any(sub_image.size > 0 for sub_image in sub_images):
            if len(sub_images) > 0 and len(sub_images[0].shape) == 3:
                sub_height, sub_width, channels = sub_images[0].shape
                height = sub_height * self.row
                width = sub_width * self.col
                combined_image = np.zeros((height, width, channels), dtype=np.uint8)

                idx = 0
                for row in range(self.row):
                    for col in range(self.col):
                        combined_image[row * sub_height: (row + 1) * sub_height,
                                    col * sub_width: (col + 1) * sub_width, :] = sub_images[idx]
                        idx += 1

                self.image[self.y:self.y+self.h, self.x:self.x+self.w] = combined_image     
        return self.image
    
    def rotate_image_if_needed(self):
        image = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        # image size
        height, width = image.shape[:2]
        # 가로 세로 확인 후 rotate 90 degreese
        if width > height:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            rotated_image = image 
        return rotated_image
