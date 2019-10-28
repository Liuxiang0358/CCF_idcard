#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:00:36 2019

@author: lx
"""
dict_ = {'壮': '6',
 '藏': '17',
 '裕固': '12,8',
 '彝': '18',
 '瑶': '14',
 '锡伯': '13,7',
 '乌孜别克': '4,7,7,7',
 '维吾尔': '11,7,5',
 '佤': '6',
 '土家': '3,10',
 '土': '3',
 '塔塔尔': '12,12,5',
 '塔吉克': '12,6,7',
 '水': '4',
 '畲': '12',
 '撒拉': '15,8',
 '羌': '7',
 '普米': '12,6',
 '怒': '9',
 '纳西': '7,6',
 '仫佬': '5,8',
 '苗': '8',
 '蒙古': '13,5',
 '门巴': '3,4',
 '毛南': '4,9',
 '满': '13',
 '珞巴': '10,4',
 '僳僳': '14,14',
 '黎': '15',
 '拉祜': '8,9',
 '柯尔克孜': '9,5,7,7',
 '景颇': '12,11',
 '京': '8',
 '基诺': '11,10',
 '回': '6',
 '赫哲': '14,10',
 '哈萨克': '9,11,7',
 '哈尼': '9,5',
 '仡佬': '5,8',
 '高山': '10,3',
 '鄂温克': '11,12,7',
 '俄罗斯': '9,8,12',
 '鄂伦春': '11,6,9',
 '独龙': '9,5',
 '东乡': '5,3',
 '侗': '8',
 '德昂': '15,8',
 '傣': '12',
 '达斡尔': '6,14,5',
 '朝鲜': '12,14',
 '布依': '5,8',
 '保安': '9,6',
 '布朗': '5,10',
 '白': '5',
 '阿昌': '7,8',
 '汉': '5'}
import check.check_number as number 
import check.language_deal as address
import re
import difflib
#import numpy as np
def ischinese(char):
    ch = ''
    for c in char:
        if u'\u4e00' <= c <= u'\u9fff':
            ch = ch + c
    return ch
    
def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()
def get_stroke(c):
    # 如果返回 0, 则也是在unicode中不存在kTotalStrokes字段
    strokes_path = 'addrs_libs/strokes.txt'
    strokes = []
    with open(strokes_path, 'r') as fr:
        for line in fr:
            strokes.append(int(line.strip()))

    unicode_ = ord(c)

    if 13312 <= unicode_ <= 64045:
        return strokes[unicode_-13312]
    elif 131072 <= unicode_ <= 194998:
        return strokes[unicode_-80338]
    else:
        print("c should be a CJK char, or not have stroke in unihan data.")
        # can also return 0

def check_race(race):
#    race = '薇'
    line = ''
    for l in race:
        line = '' + str(get_stroke(l))  + ','
    line = line[:-1]
    values  = list(dict_.values())
    distance = []
    for v in values:
        distance.append(string_similar(line, v))
    m = max(zip(distance,dict_.keys()))
    return m[1]
#    values[distance.index(max(distance))]
    
def check(dic,f):
#    result = {'address':'','sex':'','birth':'','number':'','name':'','jiguan':'南冲市公安局','riqi':'2018.03.20-2028.03.20'}
    sex = {'0':'女','1':'男'}
    dice = dic
    dice_new = {}
    if f:
        if dice['name'] != '':
            dice_new['姓名'] = ischinese(dice['name'])
        else:
            dice_new['姓名'] = '宋非凡'
        if dice['sex'] !='' and dice['sex'] in ['男','女']:
                dice_new['性别'] = dice['sex']          
        else:
                dice_new['性别'] = '男' 
        if dice['race'] !='': 
            if dice['race'] in dict_.keys():
                dice_new['民族'] = dice['race'] 
            else:
                dice_new['民族'] = check_race(dice['race'] )
        else:
                dice_new['民族'] = '汉' 
        if dice['address'] !='':
                dice_new['地址'] = address.address(dice['address'])        
        else:
                dice_new['地址'] = '黑龙江省鸡西市虎林市虎林镇于林村'
        if dice['number'] !='':
                dice_new['身份证号码'] =      dice['number']   
        else:
                dice_new['身份证号码'] = '230381199209286096'
        if dice['birth'] !='':
                dice_new['出生日期'] =      dice['birth']   
        else:
                dice_new['出生日期'] = ['1992','9','28']
        return dice_new
                
    else:
        if dice['jiguan'] != '':
            dice_new['机关'] = dice['jiguan']
        else:
            dice_new['机关'] = '宋非凡'
        if dice['riqi'] !='' :
                dice_new['日期'] = dice['riqi']          
        else:
                dice_new['日期'] = '2018.12.08-2038.12.08'
        return dice_new
