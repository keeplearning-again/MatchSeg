import pandas as pd
import shutil
import os
import numpy as np

df = pd.read_csv('ISIC2018_Task3_Training_GroundTruth.csv')

def one_hot_to_category(one_hot):
    return np.argmax(one_hot)

category_names = {0:'MEL', 1:'NV',2:'BCC',3:'AKIEC',4:'BKL',5:'DF',6:'VASC'}




for index, row in df.iterrows():
    img_name = row['image']
    img_label = one_hot_to_category(row[1:].values)
    img_label = category_names[img_label]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join('HAM10000\images', img_name + '.jpg')
    img_path = os.path.join(current_dir, relative_path)
    seg_img_path = os.path.join('HAM10000\masks', img_name + '_segmentation.png')
    img_dest_path = os.path.join('HAM10000', img_label, 'images')
    seg_img_dest_path = os.path.join('HAM10000', img_label, 'masks')
    
    if not os.path.exists(img_dest_path):
        os.makedirs(img_dest_path)
    if not os.path.exists(seg_img_dest_path):
        os.makedirs(seg_img_dest_path)
    
    img_dest = os.path.join(img_dest_path, img_name + '.jpg')
    seg_img_dest = os.path.join(seg_img_dest_path, img_name + '.png')
    shutil.move(img_path, img_dest)
    shutil.move(seg_img_path, seg_img_dest)  

