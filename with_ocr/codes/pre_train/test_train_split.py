

'''

/wrong_json_img has files ,
filter for .jpg files 
split those file in two folders train , test folders using test_ratio
copy images from wrong_json_img to train and test folders
'''

import os 
import shutil
import numpy as np
import argparse

def get_files_from_folder(path):
      files = os.listdir(path)
      _file = [f for f in files if f.endswith('.JPG')]
      return np.asarray(_file)


def main(path_to_data, path_to_test_data, train_ratio):
    files = get_files_from_folder(os.path.join(path_to_data))
    
    train_files = files[:int(len(files)*train_ratio)]
    test_files = files[int(len(files)*train_ratio):]

    if not os.path.exists(os.path.join(path_to_test_data,'train')):
        os.makedirs(os.path.join(path_to_test_data,'train'))

    if not os.path.exists(os.path.join(path_to_test_data,'test')):
        os.makedirs(os.path.join(path_to_test_data,'test'))

    for file_name in train_files:
        dst = os.path.join(path_to_test_data,'train',file_name)
        src = os.path.join(path_to_data,file_name)
        shutil.copy(src,dst)

    for file_name in test_files:
        dst = os.path.join(path_to_test_data,'test',file_name)
        src = os.path.join(path_to_data,file_name)
        shutil.copy(src,dst)

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset divider")
    parser.add_argument("--data_path", required=False,\
                            default=r'data/phase1',\
                            help="Path to data")
      
    parser.add_argument("--test_data_path_to_save",required=False,\
                            default=r'data',\
                            help="Path to test data where to save")
      
    parser.add_argument("--train_ratio", required=False,default = 0.8,\
        help="Train ratio - 0.7 means splitting data in 80 % train and 20 % test")
      
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.data_path, args.test_data_path_to_save, float(args.train_ratio))