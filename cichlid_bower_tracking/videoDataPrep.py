import subprocess, os, pdb, datetime, sys
from helper_modules.file_manager import FileManager as FM
import tarfile
import pathlib
import pandas as pd
import shutil
from math import ceil
import random
# localDir = os.getenv('HOME').rstrip('/') + '/' + 'Temp/CichlidAnalyzer/'
# output = subprocess.run(['rclone', 'copy', "p_dropbox:/CoS/BioSci/BioSci-McGrath/Apps/CichlidPiData/__AnnotatedData/LabeledVideos/Clips", "C:/Users/prera/OneDrive/Desktop/McGrath Lab/videoData"], capture_output = True, encoding = 'utf-8')
# print(output)

directory =  "C:/Users/prera/OneDrive/Desktop/McGrath Lab/videoData"

csv_path = "C:/Users/prera/OneDrive/Desktop/McGrath Lab/ManualLabels.csv"
labels_df = pd.read_csv(csv_path)
# print(labels_df.head())
labels = ['c','f','p','t','b','m','s','x','o','d']
labels_df.set_index('ClipName',inplace=True)
# # print(labels_df.index)
# print('MC16_2__0001_vid__292__10672__1014__350__1109' in labels_df.index)
# print(labels_df.loc['MC16_2__0001_vid__292__10672__1014__350__1109', 'ManualLabel'])

def createLabelFolders(labels,directory, labels_df):

    for i in labels:
        if not os.path.exists(directory+'/'+i):
            os.makedirs(directory+'/'+i)

    for filename in os.listdir(directory):
        # print(filename)
        if (pathlib.Path(filename).suffix == '.tar'):
            tar_file = os.path.join(directory,filename)
            print(tar_file)
            # pdb.set_trace()
            with tarfile.open(tar_file, 'r:*') as tar:
                extract_path = "C:/Users/prera/OneDrive/Desktop/McGrath Lab/videoData/"+ filename.split('.')[0]
                print(extract_path)
                # pdb.set_trace()

                if not os.path.exists(extract_path):
                    os.makedirs(extract_path)

                # pdb.set_trace()
                # print(filename.split('.')[0])
                tar.extractall(path = extract_path)


                video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]  # Add more as needed
                video_folder = extract_path+'/'+filename.split('.')[0]
                count = 0
                for file in os.listdir(video_folder):
                    if os.path.isfile(os.path.join(video_folder, file)):
                        if any(file.endswith(ext) for ext in video_extensions):
                            if(file[0]=="."):
                                continue
                            # pdb.set_trace()
                            if file in labels_df.index:
                                folder = labels_df.loc[file.split('.')[0],'ManualLabel']
                                # pdb.set_trace()
                                shutil.copy(video_folder+'/'+file, directory+'/'+folder)
                            else:
                                print("Key not found")

                            
def createTrainTestFolders(source_path,train_path,val_path, test_path,train_ratio,val_ratio,labels):

    for label in labels:
        label_source_path = os.path.join(source_path, label)
        label_train_path = os.path.join(train_path, label)
        label_val_path = os.path.join(val_path, label)
        label_test_path = os.path.join(test_path, label)


        os.makedirs(label_train_path, exist_ok=True)
        os.makedirs(label_val_path, exist_ok=True)
        os.makedirs(label_test_path, exist_ok=True)

        videos = os.listdir(label_source_path)
        random.shuffle(videos)

        train_count = ceil(len(videos) * train_ratio)
        val_count = ceil(len(videos) * val_ratio)

        train_videos = videos[:train_count]
        val_videos = videos[train_count:train_count+val_count]
        test_videos = videos[train_count+val_count:]

        # Copy videos to train directory
        for video in train_videos:
            shutil.copy2(os.path.join(label_source_path, video), label_train_path)
        
        for video in val_videos:
            shutil.copy2(os.path.join(label_source_path, video), label_val_path)

        # Copy videos to test directory
        for video in test_videos:
            shutil.copy2(os.path.join(label_source_path, video), label_test_path)

        print("Processed Label", label)


train_dir = "C:/Users/prera/OneDrive/Desktop/McGrath Lab/TrainTestData/train"
val_dir = "C:/Users/prera/OneDrive/Desktop/McGrath Lab/TrainTestData/val"
test_dir = "C:/Users/prera/OneDrive/Desktop/McGrath Lab/TrainTestData/test"

createTrainTestFolders(directory,train_dir, val_dir,test_dir, 0.64,0.16,labels)