import torch
import torchvision
import json
import urllib
import datetime
import os
import time
import warnings
# import datasets
# import presets
import torch
import torch.utils.data
import torchvision
import torchvision.datasets.video_utils
# import utils
import pdb
from torch import nn
# from torch.utils.data.dataloader import default_collate
from torchvision.datasets.samplers import DistributedSampler, RandomClipSampler, UniformClipSampler
import tqdm
import torchvision.transforms as T
from torchvision.io import read_video
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


# class CustomVideoDataset(Dataset):
#     def __init__(self, video_paths, transform=None):
#         self.video_paths = video_paths
#         self.transform = transform

#     def __len__(self):
#         return len(self.video_paths)

#     def __getitem__(self, idx):
#         frames, _, _ = read_video(self.video_paths[idx])
#         if self.transform:
#             pdb.set_trace()
#             frames = torch.stack([self.transform(frame) for frame in frames])
#         return frames

# transform = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# pdb.set_trace()
# video_paths = ["C:/Users/prera/OneDrive/Desktop/McGrath Lab/0001_vid/0001_vid/0001_vid__9__822__55__622__872.mp4", "C:/Users/prera/OneDrive/Desktop/McGrath Lab/0001_vid/0001_vid/0001_vid__21__1321__109__527__868.mp4"]  # List of video paths
# pdb.set_trace()
# dataset = CustomVideoDataset(video_paths)
# frames = dataset.__getitem__(0)
# pdb.set_trace()
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)




import os
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list 
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            directory (str): The path to the directory containing the train/val/test datasets
            mode (str, optional): Determines which folder of the directory the dataset will read from. Defaults to 'train'. 
            clip_len (int, optional): Determines how many frames are there in each clip. Defaults to 8. 
        """

    def __init__(self, directory, mode='train', clip_len=8):
        folder = Path(directory)/mode  # get the directory of the specified split

        self.clip_len = clip_len

        # the following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128  
        self.resize_width = 171
        self.crop_size = 112

        # obtain all the filenames of files inside all the class folders 
        # going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)     

        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label:index for index, label in enumerate(sorted(set(labels)))} 
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)        

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        buffer = self.loadvideo(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer = self.normalize(buffer)

        return buffer, self.label_array[index]    
        

    def loadvideo(self, fname):
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))

        count = 0
        retaining = True

        # read in each frame, one at a time into the numpy buffer array
        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # will resize frames if not already final size
            # NOTE: strongly recommended to resize them during the download process. This script
            # will process videos of any size, but will take longer the larger the video file.
            if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            buffer[count] = frame
            count += 1

        # release the VideoCapture once it is no longer needed
        capture.release()

        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        buffer = buffer.transpose((3, 0, 1, 2))

        return buffer 
    
    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[1] - clip_len)
        # randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[2] - crop_size)
        width_index = np.random.randint(buffer.shape[3] - crop_size)

        # crop and jitter the video using indexing. The spatial crop is performed on 
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[:, time_index:time_index + clip_len,
                        height_index:height_index + crop_size,
                        width_index:width_index + crop_size]

        return buffer                

    def normalize(self, buffer):
        # Normalize the buffer
        # NOTE: Default values of RGB images normalization are used, as precomputed 
        # mean and std_dev values (akin to ImageNet) were unavailable for Kinetics. Feel 
        # free to push to and edit this section to replace them if found. 
        buffer = (buffer - 128)/128
        return buffer

    def __len__(self):
        return len(self.fnames)
    

source_directory= "C:/Users/prera/OneDrive/Desktop/McGrath Lab/TrainTestData"
data = VideoDataset(source_directory,'train')
num_classes = 10
device = torch.device('cpu')
model = torchvision.models.video.r2plus1d_18()
model.fc = nn.Linear(model.fc.in_features, num_classes) 
model.to(device)
num_epochs  = 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_dataloader = DataLoader(VideoDataset(source_directory), batch_size=10, shuffle=True, num_workers=0)
val_dataloader = DataLoader(VideoDataset(source_directory, mode='val'), batch_size=14, num_workers=0)
dataloaders = {'train': train_dataloader, 'val': val_dataloader}

dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

# saves the time the process was started, to compute total time at the end
start = time.time()
epoch_resume = 0
save = True
save_path = "C:/Users/prera/OneDrive/Desktop/McGrath Lab/models/model.pth"
pdb.set_trace()
os.makedirs(os.path.dirname(save_path), exist_ok=True)


def main():
    for epoch in tqdm.tqdm(range(epoch_resume, num_epochs), unit="epochs", initial=epoch_resume, total=num_epochs):
            # each epoch has a training and validation step, in that order
            for phase in ['train', 'val']:

                # reset the running loss and corrects
                running_loss = 0.0
                running_corrects = 0

                # set model to train() or eval() mode depending on whether it is trained
                # or being validated. Primarily affects layers such as BatchNorm or Dropout.
                if phase == 'train':
                    # scheduler.step() is to be called once every epoch during training
                    scheduler.step()
                    model.train()
                else:
                    model.eval()


                for inputs, labels in dataloaders[phase]:
                    # move inputs and labels to the device the training is taking place on
                    labels = labels.long()
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    # keep intermediate states iff backpropagation will be performed. If false, 
                    # then all intermediate states will be thrown away during evaluation, to use
                    # the least amount of memory possible.
                    with torch.set_grad_enabled(phase=='train'):
                        outputs = model(inputs)
                        # we're interested in the indices on the max values, not the values themselves
                        _, preds = torch.max(outputs, 1)  
                        loss = criterion(outputs, labels)

                        # Backpropagate and optimize iff in training mode, else there's no intermediate
                        # values to backpropagate with and will throw an error.
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()   

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f"{phase} Loss: {epoch_loss} Acc: {epoch_acc}")

        # save the model if save=True
            if save:
                torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': epoch_acc,
                'opt_dict': optimizer.state_dict(),
                }, save_path)

        # print the total time needed, HH:MM:SS format
            time_elapsed = time.time() - start    
            print(f"Training complete in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")


if __name__ == '__main__':
    main()