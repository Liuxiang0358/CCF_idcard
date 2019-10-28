from PIL import Image
from crnn.crnn_torch import crnnOcr
import cv2
#from crnn.crnn_torch import crnnOcr_number

#def Crnn(fold):
#    fold = fold
##    fold = '/home/lx/Downloads/ID_programmer/result'
#    t = time.time()
#    
#    for file in os.listdir(fold):
#    #    print(file)
#        for image in  os.listdir(os.path.join(fold,file)):
#            image = os.path.join(fold,file, image)
#    #        print(image)
#            img = Image.open(image).convert('L')
#            res = crnnOcr(img)
#            print(file,res)
#    print(time.time() - t)# -*- coding: utf-8 -*-

def Crnn(region):
#    fold = fold
#    fold = '/home/lx/Downloads/CRNN/result/06a6f8cf-7f07-4fe6-b49b-8072c1bcff87id-face-img/'
#    t = time.time()
    result = {'address':'','sex':'','race':'','birth':'','number':'','name':'','jiguan':'','riqi':''}
#    file = os.listdir(fold)
    file = sorted(region.keys())
    for f in file:
#            f = file[5]
#            image = os.path.join(fold,f)
    #        print(image)
            if region[f].any() :
                r = region[f]
                img = Image.fromarray(cv2.cvtColor(r,cv2.COLOR_BGR2RGB)).convert('L')
#            img = Image.fromarray(region[f]).convert('L')
#            img = Image.open(image).convert('L')
#            if f in ['sex','number','riqi']:
#                res = crnnOcr_number(img)
#            else:
                res = crnnOcr(img)
            else:
                res = ''
#            print(res)
#            for key in result.keys():
#                if image.split('/')[1][2:-4] in key:
#                    result[ key]  = result[ key] + res
            for key in result.keys():
                if f[2:] in key:
                    result[ key]  = result[ key] + res
    return result
#    print(result)
#    print(time.time() - t)# -*- coding: utf-8 -*-

#image.split('/')[1][2:-5] in result.keys()
#str(result.keys()).find(image.split('/')[1][2:-5])
