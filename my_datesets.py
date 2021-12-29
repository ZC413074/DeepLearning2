import os
import cv2
import glob
import torch
import logging
import argparse
import numpy as np
from PIL import Image, ExifTags
from pathlib import Path
from itertools import repeat
import xml.etree.ElementTree as ET
from multiprocessing.pool import ThreadPool
from torch.utils.data.dataset import Dataset

image_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vidio_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass
    return s

def sparse_xml(input_file, classes):
    input_file = open(input_file, encoding='utf-8')
    tree=ET.parse(input_file)
    root = tree.getroot()
    size = root.find('size')
    w = float(size.find('width').text)
    h = float(size.find('height').text)
    labels=[]
    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = [str(cls_id), str(float(xmlbox.find('xmin').text)/w), str(float(xmlbox.find('ymin').text)/h), str(float(xmlbox.find('xmax').text)/w), str(float(xmlbox.find('ymax').text)/h)]
        labels.append(b)
    return np.array(labels, dtype=np.float32)

def voc_img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'JPEGImages' + os.sep, os.sep + 'Annotations' + os.sep  # /images/, /labels/ substrings
    return [x.replace(sa, sb, 1).replace('.' + x.split('.')[-1], 'xml') for x in img_paths]

# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized

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
    with open(path) as f:
        file_names = f.readlines()
    for i,file_name in enumerate(file_names) :
        if i==0:
            VOCdevkit_path=file_name.strip()
            continue
        image_file=os.path.join(VOCdevkit_path, 'VOC%s\\JPEGImages\\%s.jpg'%(year, file_name.strip().split()[0]))
        label_file=os.path.join(VOCdevkit_path, 'VOC%s\\Annotations\\%s.xml'%(year, file_name.strip().split()[0]))
        if not (os.path.exists(image_file) and os.path.exists(label_file)):
            print(f"{prefix} : \n {image_file} or {label_file} does not exist")
            continue
        image_files.append(image_file)
        label_files.append(label_file)
    assert image_files, f'{prefix}:No images or labels were founded'
    image_files.sort()
    label_files.sort()
    return image_files, label_files

class VOCDataset(Dataset):
    def __init__(self, path, net_img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        super(VOCDataset, self).__init__()
        self.net_img_size = net_img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-net_img_size // 2, -net_img_size // 2]
        self.stride = stride
        self.path = path
        # step1 read image file and label files 
        self.image_files, self.label_files = load_voc_detection_data(self.path)

        # step2 load labels to cache
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files + self.image_files) or 'results' not in cache:  # changed
                cache = self.cache_labels_write(cache_path, prefix)  # re-cache
        else:
            cache = self.cache_labels_write(cache_path, prefix)  # cache

        # step3 read labels、label files、shapes、shape files from cache
        cache.pop('hash')  # remove hash
        labels, shapes = zip(*cache.values())
        self.labels = list(labels[:-2])
        self.shapes = np.array(shapes[:-2], dtype=np.float64)
        self.image_files = list(cache.keys())  # update
        self.label_files = voc_img2label_paths(cache.keys())  # update

        # step4 read images
        n=len(self.image_files)
        self.imgs = [None] * n
        if cache_images: #
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 8 threads
            for i, x in enumerate(results):
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'

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

    def cache_labels_write(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing、found、empty、duplicate
        images_labels = zip(self.image_files, self.label_files)
        for i, (im_file, lb_file) in enumerate(images_labels):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                if(i==len(self.image_files)-1):
                    print(i)
                shape = exif_size(im)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in image_formats, f'invalid image format {im.format}'
                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    l=[]
                    if(lb_file.endswith("txt")):
                        with open(lb_file, 'r') as f:
                            l = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
                    elif(lb_file.endswith("xml")):
                        l=sparse_xml(lb_file, self.hyp.classes)
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        #assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape]
            except Exception as e:
                nc += 1
                print(f'{prefix}:WARNING: Ignoring corrupted image and/or label {im_file}: {e}')
        if nf == 0:
            print(f'{prefix}:WARNING: No labels found in {path}.')

        x['hash'] = get_hash(self.label_files + self.image_files)
        x['results'] = [nf, nm, ne, nc, i + 1]
        torch.save(x, path)  # save for next time
        logging.info(f'{prefix}New cache created: {path}')
        return x

    def cache_labels_read(self, path=Path('./labels.cache'), prefix=''):
        cache.pop('hash')  # remove hash
        labels, shapes = zip(*cache.values())
        return  labels, shapes

if __name__ == '__main__':
    path=r"train.txt"
    imgsz=int(640)
    batch_size=int(16)
    augment=True
    parser = argparse.ArgumentParser()
    hyp = parser.parse_args()
    hyp.classes=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplan','sheep','sofa','train','tvmonitor']
    rect=False
    cache_images=True
    single_cls=False
    stride=32
    pad=0.0
    image_weights=False
    prefix="train"
    VOCDataset(path,imgsz,batch_size,augment,hyp,rect,cache_images,single_cls,stride,pad,image_weights,prefix)
