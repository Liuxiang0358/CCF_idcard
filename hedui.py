#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 22:28:31 2019

@author: lx
"""

import csv 

def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False
def clear_ch(char):
    a = ''
    for i in range(len(char)):
        if is_Chinese(char[i]):
            a =  a + char[i]
    return a
def keep_shuzi(char):
    figure = '0123456789'
    a = ''
    for i in range(len(char)):
        if char[i] in figure:
            a =  a + char[i]
    return a
def keep_shuzi1(char):
    figure = 'X0123456789'
    a = ''
    for i in range(len(char)):
        if char[i] in figure:
            a =  a + char[i]
    return a
def keep_shuzi2(char):
    figure = '0123456789.-长期'
    a = ''
    for i in range(len(char)):
        if char[i] in figure:
            a =  a + char[i]
    return a


csv_file=csv.reader(open('test.csv','r',encoding='utf-8-sig'))
#print(csv_file) #可以先输出看一下该文件是什么样的类型

content=[] #用来存储整个文件的数据，存成一个列表，列表的每一个元素又是一个列表，表示的是文件的某一行

for line in csv_file:
#    print(line) #打印文件每一行的信息
    content.append(line)
    
csv_file=csv.reader(open('submit_example.csv','r',encoding='utf-8-sig'))
#print(csv_file) #可以先输出看一下该文件是什么样的类型

content_example=[] #用来存储整个文件的数据，存成一个列表，列表的每一个元素又是一个列表，表示的是文件的某一行

for line in csv_file:
#    print(line) #打印文件每一行的信息
    content_example.append(line)
a = []
for con in content:
   a.append(con[0])
    
for con in content_example:
   if con[0] not in  a:
       content.append(con)
      
#for con in content:
#  for c in con:
#      if ',' in c or c == '' :
#          
#         print(con[0])
#         print('1')
#      if len(con) != 11:
#          print('0')
for i in range(5000 - len(content)):
    content.append(content[16])
for contex in content:
    for idx,text in enumerate(contex):
        if idx == 1 or idx == 2 or idx == 3 or idx == 7 or idx == 9:
            contex[idx] = clear_ch(contex[idx])
            if contex[idx] == '' or contex[idx] == '0':
                if idx ==3:
                    contex[idx] = '女'
                if idx ==2:
                    contex[idx] = '汉'
                if idx ==7:
                    contex[idx] = '浙江省杭州市余杭区塘栖镇泰山村'
                if idx ==9:
                    contex[idx] = '杭州市余杭区公安局'
                if idx ==1:
                    contex[idx] = '宋非凡'
        if idx == 4 or idx == 5 or idx == 6:
            contex[idx] = keep_shuzi(contex[idx])
            if contex[idx] == '' or contex[idx] == '0':
                if idx ==4:
                    contex[idx] = '1961'
                if idx ==5:
                    contex[idx] = '5'
                if idx ==6:
                    contex[idx] = '15'
                
        if idx == 8:
            contex[idx] = keep_shuzi1(contex[idx])
            if contex[idx] == ''  or contex[idx] == '0':
                contex[idx] = '51070319610715654'
        if idx == 10:
            contex[idx] = keep_shuzi2(contex[idx])
            if contex[idx] == ''  or contex[idx] == '0':
                contex[idx] = '2017.06.24-2027.06.24'


with open("test1.csv","w",encoding='utf-8-sig') as csvfile: 
    writer = csv.writer(csvfile)
 
    #先写入columns_name
    for res in content:
        writer.writerow(res)