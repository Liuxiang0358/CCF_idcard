# -*- coding: utf-8 -*-

import aligned
import detect_id
import  crnn_dic
import os
from matplotlib import pyplot as plt
import cv2
import check_total
import csv
#application = create_app()  
if __name__ == '__main__':
    result = []
    fold1 = '/home/lx/Downloads/BDCI/data/Test'    ############  测试文件路径
    fold  = os.listdir(fold1)
#    sort = sorted(sort)
    for i,file in  enumerate(fold):
#        try :
#            i=  559#             file_name = os.path.join(fold,sort[2558])  ####
#             print(file)
             file_name = os.path.join(fold1,file)
    #             file= '0ae9a09fda754fa191d9457e3243cdcb.jpg'
             img = cv2.imread(file_name)
           
             img_face = img[:400,:,:]
             img_back = img[400:,:,:]
    #         plt.imshow(img_back)
    #         plt.imshow(img_face)
             ################
#             file = 1
             f =True
             align_img,box = aligned.first( img_face,f)
             region  =  detect_id.detect_third(align_img,box,f)
    #            plt.imshow(align_img)
    #            plt.imshow(region['p_birth'])
    #            plt.imshow(region['p1address'])
    #            plt.imshow(region['p2address'])
    #            plt.imshow(region['p_name'])
    #            plt.imshow(region['p_number'])
    #            plt.imshow(region['p_sex'])
             
             res = crnn_dic.Crnn(region)
             res_face = check_total.check(res,f)
#             res_face = res
             
#             file_name = os.path.join(fold,sort[2*i])
    #             file= '0ae9a09fda754fa191d9457e3243cdcb.jpg'
#             img_back = cv2.imread(file_name)
             f = False
             align_img,box = aligned.first( img_back,f)
             region  =  detect_id.detect_third(align_img,box,f)
            #    time_Take1 = time.time()
    #         plt.imshow(region['p_jiguan'])
#    plt.imshow(region['p_riqi'])
             res = crnn_dic.Crnn(region)
             res_back = check_total.check(res,f)
#             res_back = res
#                 print(res_back)
#            plt.imshow(align_img)
             if len(res_face['出生日期']) != 3:
                 res_face['出生日期'] = ['0','0','0']
             res_new = {}
             res_new.update(res_face)
             res_new.update(res_back)
             res_new['file'] = file[:-4]
             result.append(res_new)
             print(i)
#        except:
#             print('False')




#############  
order = ['file','姓名','民族','性别','出生日期','地址','身份证号码','机关','日期']
with open("test.csv","w",encoding='utf-8-sig') as csvfile: 
    writer = csv.writer(csvfile)
 
    #先写入columns_name
    for res in result:
        line = []
        for ort in order:
            if ort == '出生日期':
                line.append(res[ort][0])
                line.append(res[ort][1])
                line.append(res[ort][2])
            else:
                if ort in res.keys():
                    line.append(res[ort])
                else:
                    line.append('')
#    writer.writerow(["index","a_name","b_name"])
        writer.writerow(line)
#    写入多行用writerows
#    writer.writerows([[0,1,3],[1,2,3],[2,3,4]])
