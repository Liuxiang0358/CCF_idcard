# -*- coding: utf-8 -*-

import aligned
import second
import detect_id
import  crnn_dic
import time
import os
from matplotlib import pyplot as plt
import cv2
import check_total
#application = create_app()  
  
if __name__ == '__main__':
    
     file_name = os.path.join('data/Test',file)
#             file_name = '0ae9a09fda754fa191d9457e3243cdcb.jpg'
     img = cv2.imread(file_name)
     img_face = img[:400,:,:]
     img_back = img[400:,:,:]
#             plt.imshow(img_back)
     ################
     img_face = cv2.cvtColor(img_face ,cv2.COLOR_BGR2RGB)
     box = second.detect_second(img_face)
     bbox = [b[0] for b in box]
     f =True
#     if 'face' in file_name:
#         f = True
     if '5' in bbox:
        img_face = aligned.rotate_bound(img_face,180)
     align_img,box = aligned.first( img_face,f)
     region  =  detect_id.detect_third(align_img,box,f )
#             plt.imshow(align_img)
#             plt.imshow(region['p_birth'])
#             plt.imshow(region['p_birth'])
     res = crnn_dic.Crnn(region)
     res_face = check_total.check(res)
     
#             print(res_face)
     
     ##################
     img_back = cv2.cvtColor(img_back ,cv2.COLOR_BGR2RGB)
     box = second.detect_second(img_back)
     bbox = [b[0] for b in box]
     f = False
     if '5' in bbox:
        img_back = aligned.rotate_bound(img_back,180)
     align_img,box = aligned.first( img_back,f)
     region  =  detect_id.detect_third(align_img,box,f)
    #    time_Take1 = time.time()
#             plt.imshow(img_back)
     res = crnn_dic.Crnn(region)
     res_back = check_total.check(res)
#             print(res_back)
     if len(res_face['出生日期']) != 3:
         res_face['出生日期'] = ['0','0','0']
     res_new = {}
     res_new.update(res_face)
     res_new.update(res_back)
     res_new['file'] = file[:-4]
     result.append(res_new)
