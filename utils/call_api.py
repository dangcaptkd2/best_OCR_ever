import requests
import json
   
API_ENDPOINT = "http://127.0.0.1:3050/banner_detection/check_ocr"
#API_ENDPOINT = "http://localhost:3050/banner_detection/check_ocr"



def call(name: str, path: str) -> dict:
    ##### Truyền path ảnh vào đây
    #### format: 
    """
    data = {
        'file': ('tên_ảnh.jpg', open('path_ảnh.jpg', 'rb'))
    }
    """
    data = {
        'file': (name, open(path, 'rb'))
    }
    r = requests.post(url = API_ENDPOINT, files = data)
    print(r.text)
    result = json.loads(r.text)
    return result

path = '/home/quyennt72/banner_checking_fptonline/tmp_images/banner_sexy_1607/'
import os
c = 0
import time
start = time.time()
for name in os.listdir(path):
    result = call(name, path+name)
    c+=1
    print(c)
end = time.time()
ti = (end-start)/len(os.listdir(path))
print(">>>>>>>>>>", ti)