import cv2
import glob
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data.dataset import Dataset

image_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vidio_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

def load_voc_detection_data(path, year=2007, prefix=''):
    '''
    path: the path of txt, which local files names are here 
    -------VOCDATA SET------
    VOCdevkit  VOCdevkit_path   local or global is ok
        |---VOCcode
            ...
        |---VOC2007
            |--Annotations        *.xml   label files
            |--ImageSets          *.txt   
               |--Layout          test.txt train.txt trainval.txt val.txt which can be read for trian or test classify
               |--Main            single class txt or others
               |--Segmentation    test.txt train.txt trainval.txt val.txt which can be read for trian or test segmentation
            |--JPEGImages         *.png   iamge files
            |--SegmentationClass  *.png   Segmentated iamge files
            |--SegmentationObject *.png   Segmentated iamge files
        ...
    '''
    VOCdevkit_path=""
    image_files, label_files = [],[]
    with open(all_files_path) as f:
        file_names = f.readlines()
    for i,file_name in enumerate(file_names) :
        if i==1:
            VOCdevkit_path=file_name.strip()
            continue
        image_file=os.path.join(VOCdevkit_path, 'VOC%s/JPEGImages/%s.png'%(year, file_name.strip()))
        label_file=os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml'%(year, file_name.strip()))
        if (not image_file.exist()) or (not label_file.exist()):
            print(f"{prefix} : \n {image_file} or {label_file} does not exist")
            continue
        image_files.append(image_file)
        label_files.append(label_file)
    assert img_files, f'{prefix}:No images or labels were founded'
    image_files.sort()
    label_files.sort()
    return image_files, label_files

class VOCDataset(Dataset):
    def __init__(self, path, net_img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        super(YoloDataset, self).__init__()
        self.net_img_size = net_img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-net_img_size // 2, -net_img_size // 2]
        self.stride = stride
        self.path = path

        self.image_files, self.label_file = load_voc_detection_data(self.path)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        image, box  = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random = self.train)
        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box

if __name__ == '__main__':
    path=r"G:\deepdata\VOCDataset\VOC2007"
    imgsz=int(640)
    batch_size=int(16)
    augment=True
    parser = argparse.ArgumentParser()
    hyp = parser.parse_args()
    rect=False
    cache_images=False
    single_cls=False
    stride=32
    pad=0.0
    image_weights=False
    prefix="train"
    YoloDataset(path,imgsz,batch_size,augment,hyp,rect,cache_images,single_cls,stride,pad,image_weights,prefix)
