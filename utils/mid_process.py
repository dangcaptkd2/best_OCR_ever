import os
import cv2
from tqdm import tqdm
import shutil
import json
from PIL import Image

def up(a: list, b: list) -> bool:
  if a[1] >= b[3]:
    return True 
  return False

def beside(a: list, b: list) -> bool:
  if a[0] >= b[0]:
    if (b[1] <= a[1] and a[1]<=b[3]) or (b[1] <= a[3] and a[3]<= b[3]) or (a[1] <= b[1] and b[1] <= a[3]):
      return True
  return False

def swap_dic(dic, key_a, key_b):
    temp = dic[key_a]
    dic[key_a] = dic[key_b]
    dic[key_b] = temp

def check_near_box(l1, l2, thres_x=10, thres_y=5):
  x1,y1,x2,y2 = l1
  x1_,y1_,x2_,y2_ = l2

  if abs(x2-x1_)<thres_x and abs(y1-y1_)<thres_y and abs(y2-y2_)< thres_y:
    return True
  return False

def merge_2_boxes(d: dict, id1: int, id2: int) -> dict:
  x1,y1,x2,y2 = d[id1]
  x1_,y1_,x2_,y2_ = d[id2]

  new_cor = [x1,y1,x2_,y2_]
  d[id1] = new_cor
  del d[id2]
  return d

def merge(dic: dict, thres_h: float, thres_w: float) -> dict:
  d=dic.copy()
  sta = 0
  end = 1
  max_id = len(d)
  while end<max_id:
    if check_near_box(d[sta], d[end], thres_x=thres_w, thres_y=thres_h):
      d = merge_2_boxes(d, sta, end) 
      end+=1
    else:
      sta = end
      end = sta+1
  return d

# def action_merge(sorted_cor, name, image_path):

#   img = cv2.imread(image_path)
#   h,w,_ = img.shape()

#   des = './debugs/crop_images/' + name
#   if os.path.isdir(des):
#     shutil.rmtree(des)
#   os.makedirs(des)

#   new_r = merge(sorted_cor[name], h, w)
#   key = sorted(list(map(int, list(new_r.keys()))))
#   final_dict = {}
#   for idx, key in enumerate(key):
#     final_dict[idx] = new_r[key]

#   for key, box in final_dict.items():
#     crop = img[box[1]:box[3], box[0]:box[2]]
#     final_path = des + '/' + str(key) +'.jpg'
#     cv2.imwrite(final_path, crop)

def expand_bbox(bbox: list, img_w: int, img_h: int, thres: float=0.02) -> list:
  x1,y1,x2,y2 = bbox 
  new_x1 = x1-thres*img_w
  new_x2 = x2+thres*img_w

  new_y1 = y1-thres*img_h
  new_y2 = y2+thres*img_h
  return list(map(int, [new_x1, new_y1, new_x2, new_y2]))

def merge_boxes_to_line_text(img, sorted_cor, thres=0.15, expand=True):  
  h,w,_ = img.shape
  thres_h = thres*h
  thres_w = thres*w

  new_r = merge(sorted_cor, thres_h=thres_h, thres_w=thres_w)
  key = sorted(list(map(int, list(new_r.keys()))))
  final_dict = {}
  for idx, key in enumerate(key):
    final_dict[idx] = new_r[key]
  lines =[]
  for key, box in final_dict.items():
    if expand:
      new_box = expand_bbox(box, img_w=w, img_h=h)
      crop = img[new_box[1]:new_box[3], new_box[0]:new_box[2]]
    else:
      crop = img[box[1]:box[3], box[0]:box[2]]
    if crop.shape[0]>10 and crop.shape[1]>10:
      lines.append(crop)  
  return lines

def mid_process(image, result_detect):

  sorted_cor = {}

  list_key = list(result_detect.keys())
  for i in range(len(list_key)-1):
    for j in range(i+1, len(list_key)):
        if (up(result_detect[list_key[i]], result_detect[list_key[j]]) or beside(result_detect[list_key[i]], result_detect[list_key[j]])):
          swap_dic(result_detect, list_key[i], list_key[j])
  sorted_cor = result_detect

  list_arr = []
  for index, box in sorted_cor.items():
      crop = image[box[1]:box[3], box[0]:box[2]]
      list_arr.append(Image.fromarray(crop.astype('uint8'), 'RGB').convert('L'))

  return list_arr, sorted_cor