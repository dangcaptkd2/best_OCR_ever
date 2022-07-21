import cv2
from flask_restful import Resource 

from detection import DETECTION
from recognition import RECOGNITION 
from recognition_vn.vietocr import RECOGNITION_VN
from utils.mid_process import mid_process, merge_boxes_to_line_text
# from utils.policy_checking import check_text_eng, check_text_vi
from utils.utils import check_is_vn, clear_folder

import torch
import os
import time
import time

class Stat(Resource):
    def get(self):
        return dict(error=0,message="server start")

class banner_cheking():
    def __init__(self, path_image_root: str, target: str="all") -> None:
        self.path_image_root = path_image_root
        self.detect = DETECTION()
        self.recog = RECOGNITION()
        self.recog_vn = RECOGNITION_VN()
        self.target = target # all | vietnamese | english
        
    def predict(self, filename: str) -> dict:
        item = {
            'text_english': None,
            'text_vietnamese': None,
        }

        image_path = os.path.join(self.path_image_root, filename)
        name = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')    
        since = time.time() 
        img = cv2.imread(image_path)
        if img is None:
            return item 

        result_detect = self.detect.create_file_result(img, name=name)
        torch.cuda.empty_cache()
        list_arr, sorted_cor = mid_process(image = img, result_detect=result_detect)
    
        if len(list_arr)>0:
            if self.target == "all":
                result_eng = self.recog.predict_arr(bib_list=list_arr)
                torch.cuda.empty_cache()
                text_en = [result_eng[k][0] for k in result_eng if result_eng[k][1]>0.6]

                if len(text_en)==0:
                    return item
                    
                item['text_english'] = ' '.join(text_en)

                if not check_is_vn(text_en):
                    return item

                bboxs = merge_boxes_to_line_text(img, sorted_cor)
                text_vn = self.recog_vn.predict(bboxs)
                torch.cuda.empty_cache()
                item['text_vietnamese'] = ' '.join(text_vn)

            elif self.target == "vietnamese":
                bboxs = merge_boxes_to_line_text(img, sorted_cor)
                text_vn = self.recog_vn.predict(bboxs)
                torch.cuda.empty_cache()
                item['text_vietnamese'] = ' '.join(text_vn)

            elif self.target == "english":
                result_eng = self.recog.predict_arr(bib_list=list_arr)
                torch.cuda.empty_cache()
                text_en = [result_eng[k][0] for k in result_eng if result_eng[k][1]>0.6]

                if len(text_en)==0:
                    return item
                    
                item['text_english'] = ' '.join(text_en)

        clear_folder()
        return item

if __name__ == '__main__': 
    print("helllooooo")
    a = banner_cheking(path_image_root='./static/uploads/', target="english")
    r = a.predict('5.png')
    print(r)
