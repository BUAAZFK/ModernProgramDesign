import os 
import tqdm
import matplotlib.pyplot as plt
from PIL import ImageFilter, Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import numpy as np
import torchvision.transforms as T
import cv2
import torch.nn as nn



class Filter:

    def __init__(self, image, params):
        self.image = image
        self.params = params

    def filter(self):
        pass

# Implement edge extraction
class EdgeExtra(Filter):

    def __init__(self, image, params):
        super().__init__(image, params)

    def filter(self, im):
        im = im.filter(ImageFilter.FIND_EDGES)
        return im

class Sharpen(Filter):

    def __init__(self, image, params):
        super().__init__(image, params)
    
    def filter(self, im):
        im = im.filter(ImageFilter.SHARPEN)
        return im
    
class Vague(Filter):

    def __init__(self, image, params):
        super().__init__(image, params)

    def filter(self, im):
        im = im.filter(ImageFilter.BLUR)
        return im

class SizeAdjust(Filter):

    # def __init__(self, image, params):
    #     super().__init__(image, params)

    def filter(self, im):
        im = im.resize(self.params, resample = Image.BICUBIC, box = None, reducing_gap = 5)
        # im = im.resize(self.params,Image.ANTIALIAS) # resize with high quality
        return im

class ImageEnhance:

    def GrayTranform(self, imgs):
        for i in range(len(imgs)):
            gray_img = T.Grayscale()(imgs[i])
            ax1 = plt.subplot(121)
            ax1.set_title('original')
            ax1.imshow(imgs[i])
            ax2 = plt.subplot(122)
            ax2.set_title('gray')
            ax2.imshow(gray_img,cmap='gray')
            plt.savefig(f'./Gray/{i}.png')
    
    def Normlize(self, imgs):
        for i in range(len(imgs)):
            normalized_img = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(T.ToTensor()(imgs[i]))
            normalized_img = [T.ToPILImage()(normalized_img)]
            # plt.figure('resize:128*128')
            ax1 = plt.subplot(121)
            ax1.set_title('original')
            ax1.imshow(imgs[i])
            ax2 = plt.subplot(122)
            ax2.set_title('normalize')
            ax2.imshow(normalized_img[0])
            plt.savefig(f'./Norm/{i}.png')
            plt.show()

    def Rotate(self, imgs):
        for i in range(len(imgs)):
            plt.rcParams["savefig.bbox"] = 'tight'
            rotated_imgs = [T.RandomRotation(degrees=90)(imgs[i])]
            print(rotated_imgs)
            plt.figure('resize:128*128')
            ax1 = plt.subplot(121)
            ax1.set_title('original')
            ax1.imshow(imgs[i])
            ax2 = plt.subplot(122)
            ax2.set_title('90??')
            ax2.imshow(np.array(rotated_imgs[0]))
            plt.savefig(f'./Rotate/{i}.png')

    def GaussBlur(self, imgs):
        for i in range(len(imgs)):
            plt.rcParams["savefig.bbox"] = 'tight'
            blurred_imgs = [T.GaussianBlur(kernel_size=(3, 3), sigma=sigma)(imgs[i]) for sigma in (3,7)]
            plt.figure('resize:128*128')
            ax1 = plt.subplot(131)
            ax1.set_title('original')
            ax1.imshow(imgs[i])
            ax2 = plt.subplot(132)
            ax2.set_title('sigma=3')
            ax2.imshow(np.array(blurred_imgs[0]))
            ax3 = plt.subplot(133)
            ax3.set_title('sigma=7')
            ax3.imshow(np.array(blurred_imgs[1]))
            plt.savefig(f'./GaussBlur/{i}.png')
            plt.show()

    def GaussNoise(self, imgs):
        plt.rcParams["savefig.bbox"] = 'tight'
        def add_noise(inputs, noise_factor=0.3):
            noisy = inputs + torch.randn_like(inputs) * noise_factor
            noisy = torch.clip(noisy, 0., 1.)
            return noisy
        for i in range(len(imgs)):
            noise_imgs = [add_noise(T.ToTensor()(imgs[i]), noise_factor) for noise_factor in (0.3, 0.6)]
            noise_imgs = [T.ToPILImage()(noise_img) for noise_img in noise_imgs]
            plt.figure('resize:128*128')
            ax1 = plt.subplot(131)
            ax1.set_title('original')
            ax1.imshow(imgs[i])
            ax2 = plt.subplot(132)
            ax2.set_title('noise_factor=0.3')
            ax2.imshow(np.array(noise_imgs[0]))
            ax3 = plt.subplot(133)
            ax3.set_title('noise_factor=0.6')
            ax3.imshow(np.array(noise_imgs[1]))
            plt.savefig(f'./GaussNoise/{i}.png')
            plt.show()
        
    def RandomBlock(self, imgs):
        plt.rcParams["savefig.bbox"] = 'tight'
        def add_random_boxes(img,n_k,size=64):
            h,w = size,size
            img = np.asarray(img).copy()
            img_size = img.shape[1]
            boxes = []
            for k in range(n_k):
                y,x = np.random.randint(0,img_size-w,(2,))
                img[y:y+h,x:x+w] = 0
                boxes.append((x,y,h,w))
            img = Image.fromarray(img.astype('uint8'), 'RGB')
            return img
        for i in range(len(imgs)):
            blocks_imgs = [add_random_boxes(imgs[i],n_k=10)]
            plt.figure('resize:128*128')
            ax1 = plt.subplot(131)
            ax1.set_title('original')
            ax1.imshow(imgs[i])
            ax2 = plt.subplot(132)
            ax2.set_title('10 black boxes')
            ax2.imshow(np.array(blocks_imgs[0]))
            plt.savefig(f'./RandomBlock/{i}.png')
            plt.show()


class ImageShop:

    def __init__(self, format_, file_, ImList, ImResult):
        self.format_ = format_
        self.file_ = file_
        self.ImList = ImList
        self.ImResult = ImResult

    def load_images(self):
        '''
        ??????file_????????????????????????????????????
        ????????????????????????????????????
        '''
        dirs = os.listdir(self.file_)
        for dir in dirs:
            self.ImList.append(Image.open(f'{self.file_}/{dir}'))
        self.ImResult = self.ImList

    def __batch_ps(self, Filter):
        '''
        ????????????Filter?????????
        ????????????????????????????????????????????????
        '''
        for i in range(len(self.ImList)):
            self.ImResult[i] = (Filter.filter(self.ImList[i]))

    # ?????????????????????????????????????????????
    def batch_ps(self, operation,  *args):
        '''
        operation???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????None
        *args??????????????????????????????????????????????????????operation????????????
        '''
        self.load_images()
        print(f'{operation[0]}:{operation[1]}')
        # ??????eval?????????????????????????????????????????????????????????????????????Filter??????filter??????????????????
        temp = eval(operation[0])(self.ImList[0], operation[1])
        self.__batch_ps(temp)
        for arg in args:
            print(f'{arg[0]}:{arg[1]}')
            temp = eval(arg[0])(self.ImList[0], arg[1])
            self.__batch_ps(temp)

    def display(self, row = 3, col = 3, top = 9):
        '''
        ??????3x3??????????????????????????????????????????9???????????????????????????????????????9?????????
        '''
        ImResult_temp = self.ImResult[:len(self.ImResult) if len(self.ImResult)<top else top]
        for i in range(0, len(ImResult_temp),row*col):
            for j in range(1,row*col+1):
                plt.subplot(row, col, j)
                plt.imshow(ImResult_temp[i+j-1])
            plt.show()
    
    def save(self, path):
        for i in range(len(self.ImResult)):
            self.ImResult[i].save(f'{path}/{i}.{self.format_}')

class TestImageShop:
    
    def __init__(self, format_, file_, ImList, ImResult):
        self.test = ImageShop(format_, file_, ImList, ImResult)
    def batch(self, operation, *args):
        self.test.batch_ps(operation, *args)
    def save(self, path):
        self.test.save(path)
    def display(self):
        self.test.display()

def similar(img1_path, img2_path):

    img1 = cv2.imread(img1_path)
    img1 = cv2.normalize(img1, img1, 0, 255, cv2.NORM_MINMAX)
    img1 = cv2.resize(img1, (512,512))
    img2 = cv2.imread(img2_path)
    img2 = cv2.normalize(img2, img2, 0, 255, cv2.NORM_MINMAX)
    img2 = cv2.resize(img2, (512,512))
    print(img1,img2)

    img1 = img1.transpose()
    img1=torch.Tensor(img1)
    img2 = img2.transpose()
    img2=torch.Tensor(img2)

    img1=img1.unsqueeze(0)
    img2=img2.unsqueeze(0)
    print(img1.shape)
    print(img2.shape)

    mse=nn.MSELoss()
    loss=mse(img1,img2)
    print(float(loss))
    return loss

def img_similarity(img1_path, img2_path):
    """
    # ??????????????????????????????????????????
    :param img1_path: ??????1??????
    :param img2_path: ??????2??????
    :return: ???????????????
    """
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    w1, h1 = img1.shape
    w2, h2 = img2.shape
    img1 = cv2.resize(img1, (h1, w1))
    img2 = cv2.resize(img2, (h2, w2))
    # ?????????ORB?????????
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # ????????????????????????
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # knn????????????
    matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
    # ???????????????????????????
    good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
    similary = float(len(good)) / len(matches)
    if similary > 0.3:
        print("?????????ture,????????????????????????:%s" % similary)
    else:
        print("?????????false,????????????????????????:%s" % similary)
    return similary


def main():

    params = (521, 521)
    format_ = 'Png'
    file_ = "./origin_images"
    operations = ["EdgeExtra", "Sharpen", "Vague", "SizeAdjust"]
    # path = './images'
    path = './temp'

    test = TestImageShop(format_, file_, [], [])
    # test.batch((operations[3], params), (operations[1], None), (operations[2], None))
    # test.batch((operations[3],params))
    test.batch((operations[3],params))
    test.save(path)
    test.display()

def test(filepath):
    dirs = os.listdir(filepath)
    imglist = []
    for dir in dirs:
        imglist.append(Image.open(f'{filepath}/{dir}'))
    Img = ImageEnhance()
    Img.GrayTranform(imglist)
    Img.GaussBlur(imglist)
    Img.GaussNoise(imglist)
    Img.Normlize(imglist)
    # Img.RandomBlock(imglist)
    # Img.Rotate(imglist)



if __name__=='__main__':
    # similar('./origin_images/0.jpeg','./origin_images/1.jpeg')
    # print('????????????????????????????????????')
    main()
    img_similarity('./origin_images/1.jpeg','./temp/1.Png')
    # print('???????????????????????????')
    # img_similarity('./images/4.Png','./images/4.Png')
    # test('./origin_images')