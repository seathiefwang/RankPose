from __future__ import division
import random
import os
import cv2
import math
from PIL import Image
import numpy as np
from albumentations.core.transforms_interface import to_tuple, ImageOnlyTransform, DualTransform
from albumentations.augmentations import functional as F

#os.name 。它的返回值有两种： nt 和 posix, nt 表示Windwos系操作系统， posix 代表类Unix或OS X系统。
if os.name == 'posix':
    from ctypes import *
    clib = cdll.LoadLibrary(os.path.split(os.path.realpath(__file__))[0]+'/radial_blur.so')


#生成运动模糊的卷积核和锚点 angle不能为0
def genaratePsf(length, angle):
    half = length / 2
    EPS=np.finfo(float).eps                                 
    alpha = (angle-math.floor(angle/ 180) *180) /180* math.pi
    cosalpha = math.cos(alpha)  
    sinalpha = math.sin(alpha)  
    if cosalpha < 0:
        xsign = -1
    elif angle == 90:
        xsign = 0
    else:  
        xsign = 1
    psfwdt = 1 
    #模糊核大小
    sx = math.ceil(math.fabs(length*cosalpha + psfwdt*xsign - length*EPS))
    sy = math.ceil(math.fabs(length*sinalpha + psfwdt - length*EPS))
    psf1=np.zeros((sy,sx))
     
    #psf1是左上角的权值较大，越往右下角权值越小的核。
    #这时运动像是从右下角到左上角移动
    for i in range(0,sy):
        for j in range(0,sx):
            psf1[i][j] = i*math.fabs(cosalpha) - j*sinalpha
            rad = math.sqrt(i*i + j*j) 
            if  rad >= half and math.fabs(psf1[i][j]) <= psfwdt:  
                temp = half - math.fabs((j + psf1[i][j] * sinalpha) / cosalpha)  
                psf1[i][j] = math.sqrt(psf1[i][j] * psf1[i][j] + temp*temp)
            psf1[i][j] = psfwdt + EPS - math.fabs(psf1[i][j]);  
            if psf1[i][j] < 0:
                psf1[i][j] = 0
    #运动方向是往左上运动，锚点在（0，0）
    anchor=(0,0)
    #运动方向是往右上角移动，锚点一个在右上角
    #同时，左右翻转核函数，使得越靠近锚点，权值越大
    if angle<90 and angle>0:
        psf1=np.fliplr(psf1)
        anchor=(psf1.shape[1]-1,0)
    elif angle>-90 and angle<0:#同理：往右下角移动
        psf1=np.flipud(psf1)
        psf1=np.fliplr(psf1)
        anchor=(psf1.shape[1]-1,psf1.shape[0]-1)
    elif angle<-90:#同理：往左下角移动
        psf1=np.flipud(psf1)
        anchor=(0,psf1.shape[0]-1)
    psf1=psf1/psf1.sum()
    return psf1, anchor

# 应用径向模糊， center=(x,y)中心点位置， num均值力度， type模糊类型，direction为缩放，angle为角度
def apply_radial_blur(img, center=None, num=40, type='direction'):
    height, width, channel = img.shape
    img_split = cv2.split(img)
    
    if center is None:
        center = (random.randint(0,width), random.randint(0, height))
    
    for y in range(height):
        for x in range(width):
            t0, t1, t2 = 0, 0, 0
            # R = np.linalg.norm(xxx, axis=1, keepdims=True)
            R = math.sqrt((y-center[1])**2 + (x-center[0])**2)
            angle = math.atan2((y-center[1]), (x-center[0]))
            
            for i in range(num):
                if type == 'direction':
                    tmpR = R-i if R-i>0 else 0
                else:
                    tmpR = R
                    angle += 0.01# 0.01控制变化频率，步长
                
                newX = int(tmpR * math.cos(angle) + center[0])
                newY = int(tmpR * math.sin(angle) + center[1])
                
                if(newX<0):newX = 0
                if(newX>width-1):newX = width-1
                if(newY<0):newY = 0
                if(newY>height-1):newY = height-1
                
                t0 += img_split[0][newY,newX]
                t1 += img_split[1][newY,newX]
                t2 += img_split[2][newY,newX]
                
            img[y,x,0] = t0 // num
            img[y,x,1] = t1 // num
            img[y,x,2] = t2 // num
            
    return img

def apply_motion_blur(image, count):
    """
    https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    """
    image_t = image.copy()
    imshape = image_t.shape
    size = 15
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    i = imshape[1] * 3 // 4 - 10 * count
    while i <= imshape[1]:
        image_t[:, i:, :] = cv2.filter2D(image_t[:, i:, :], -1, kernel_motion_blur)
        image_t[:, :imshape[1] - i, :] = cv2.filter2D(image_t[:, :imshape[1] - i, :], -1, kernel_motion_blur)
        i += imshape[1] // 25 - count
        count += 1
    color_image = image_t
    return color_image

def apply_resize(img, scale):
    interp = [cv2.INTER_NEAREST,cv2.INTER_AREA,cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_LANCZOS4]
    h, w, _ = img.shape
    img = cv2.resize(img, (h//scale, w//scale), interpolation=interp[random.randint(0,3)])
    img = cv2.resize(img, (h, w), interpolation=interp[random.randint(2,3)])
    return img

def gasuss_noise(image, mean=0, var=0.001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

def rotate(img, angle, interpolation, border_mode, border_value=None):
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (width, height),
                         flags=interpolation, borderMode=border_mode, borderValue=border_value)
    return img

# 残影or运动模糊
def ghosting(img, dis, alpha=0.5, angle=0, scale=0):
    dx = random.uniform(-dis, dis)
    dy = dis - abs(dx)
    dy = random.choice([-dy, dy])
    tmp1 = F.shift_scale_rotate(img, angle, scale+1, dx, dy, value=0)
    if random.random() < 0.5:
        tmp1 = F.gaussian_blur(tmp1, 3)
    tmp2 = img
    if random.random() < 0.5:
        tmp2 = F.gaussian_blur(tmp2, 3)
    
    beta = 1 - alpha
    gamma = 0
    return cv2.addWeighted(tmp1, alpha, tmp2, beta, gamma)
    

class AddSpeed(ImageOnlyTransform):
    def __init__(self, speed_coef=-1, p=.5):
        super(AddSpeed, self).__init__(p)
        assert speed_coef == -1 or 0 <= speed_coef <= 1
        self.speed_coef = speed_coef

    def apply(self, img, count=7, **params):
        return apply_motion_blur(img, count)

    def get_params(self):
        if self.speed_coef == -1:
            return {'count': int(15 * random.uniform(0, 1))}
        else:
            return {'count': int(15 * self.speed_coef)}

class Rotate(DualTransform):
    def __init__(self, limit=90, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, border_value=255, always_apply=False, p=.5):
        super(Rotate, self).__init__(always_apply, p)
        self.limit = to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.border_value = border_value

    def apply(self, img, angle=0, **params):
        return rotate(img, angle, interpolation=self.interpolation, border_mode=self.border_mode)

    def apply_to_mask(self, img, angle=0, **params):
        return rotate(img, angle, interpolation=cv2.INTER_NEAREST,
                      border_mode=cv2.BORDER_CONSTANT, border_value=self.border_value)

    def get_params(self):
        return {'angle': random.uniform(self.limit[0], self.limit[1])}

class PadIfNeededRightBottom(DualTransform):
    def __init__(self, min_height=769, min_width=769, border_mode=cv2.BORDER_CONSTANT,
                 value=0, ignore_index=255, always_apply=False, p=1.0):
        super(PadIfNeededRightBottom, self).__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.border_mode = border_mode
        self.value = value
        self.ignore_index = ignore_index

    def apply(self, img, **params):
        img_height, img_width = img.shape[:2]
        pad_height = max(0, self.min_height-img_height)
        pad_width = max(0, self.min_width-img_width)
        return np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), 'constant', constant_values=self.value)

    def apply_to_mask(self, img, **params):
        img_height, img_width = img.shape[:2]
        pad_height = max(0, self.min_height-img_height)
        pad_width = max(0, self.min_width-img_width)
        return np.pad(img, ((0, pad_height), (0, pad_width)), 'constant', constant_values=self.ignore_index)

class RandomBlur(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(RandomBlur, self).__init__(always_apply, p)
        self.blur_limit = to_tuple(15, 3)
        self.motion_limit = to_tuple(10, 3)

    def apply(self, img, blur_type, **params):
        ksize = random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2))
        # print('ksize:', ksize)
        # blur_type = random.randint(0, 90)
        # blur_type = 1
        if blur_type < 10:
            img = F.motion_blur(img, kernel=self.get_motion_kernel())
        elif blur_type < 20:
            img = F.blur(img, random.choice(range(3,16,2)))
        elif blur_type < 30:
            img = F.median_blur(img, random.choice(range(3,16,2)))
        elif blur_type < 40:
            img = F.gaussian_blur(img, ksize)
        elif blur_type < 50:
            img = ghosting(img, random.uniform(0.01, 0.1), angle=random.randint(-7, 7), scale=random.uniform(0., 0.1))
        elif blur_type < 60:
            kernel, anchor = genaratePsf(random.randint(5, 30),random.randint(-90, 90))
            img = cv2.filter2D(img, -1, kernel, anchor=anchor)
        elif blur_type < 70: # 速度会很慢
            if random.random() < 0.5:
                radial_num = random.randint(8, 30)
                radial_type = 'angle'
            else:
                radial_num = random.randint(5, 13)
                radial_type = 'direction'
            if os.name == 'posix':
                h, w, c = img.shape
                img_arr = img.ctypes.data_as(c_char_p)
                clib.apply_radial_blur(img_arr,h,w,radial_num,0 if radial_type=='angle' else 1)
            else:
                img = apply_radial_blur(img, num=radial_num, type=radial_type)
        elif blur_type < 80:
            if blur_type < 75:
                img = F.gaussian_blur(img, random.choice(range(3,9,2)))
            img = F.image_compression(img, random.randint(0,15), '.jpg')
        elif blur_type < 85:
            img = gasuss_noise(img, 0, random.uniform(0.05,0.15))
        elif blur_type < 90:
            img = apply_resize(img, random.randint(2, 7))
        else:
            img = F.gaussian_blur(img, ksize)

        end_plus = random.random()
        if end_plus < 0.05:
            img = F.gaussian_blur(img, random.choice(range(3,9,2)))
        elif end_plus < 0.1:
            kernel, anchor = genaratePsf(random.randint(3,10),random.randint(-90, 90))
            img = cv2.filter2D(img, -1, kernel, anchor=anchor)

        return img

    def get_motion_kernel(self):
        ksize = random.choice(np.arange(self.motion_limit[0], self.motion_limit[1] + 1, 2))
        assert ksize > 2
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        xs, ys = random.randint(0, ksize//2), random.randint(0, ksize - 1)
        if random.random() < 0.5:
            xs = 0
        else:
            ys = 0
        xe, ye = ksize-1 - xs, ksize-1 - ys
        
        cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)
        return kernel    

    def get_params(self):
        
        blur_type = random.randint(0, 90)

        return {'blur_type':blur_type}

def applyBlur(img):
    blur_type = random.choice([1,2,3,4,5,6,7,8,9])
    level = np.random.choice([0,1,2,3,4,5], size=2, replace=False)
    # blur_type = 1
    if blur_type == 1:
        l_value = [0,3,7,9,11,13]
        img_first = img.copy() if level[0]==0 else F.motion_blur(img, kernel=get_motion_kernel(l_value[level[0]]))
        img_second = img if level[1]==0 else F.motion_blur(img, kernel=get_motion_kernel(l_value[level[1]]))
    elif blur_type == 2:
        l_value = [0,3,5,7,9,11]
        img_first = img.copy() if level[0]==0 else F.blur(img, l_value[level[0]])
        img_second = img if level[1]==0 else F.blur(img, l_value[level[1]])
    elif blur_type == 3:
        l_value = [0,3,5,7,9,11]
        img_first = img.copy() if level[0]==0 else F.median_blur(img, l_value[level[0]])
        img_second = img if level[1]==0 else F.median_blur(img, l_value[level[1]])
    elif blur_type == 4:
        l_value = [0,5,9,11,15,17]
        img_first = img.copy() if level[0]==0 else F.gaussian_blur(img, l_value[level[0]])
        img_second = img if level[1]==0 else F.gaussian_blur(img, l_value[level[1]])
    elif blur_type == 5:
        l_value = [0,5,11,17,23,25]
        kernel_first, anchor_first = genaratePsf(l_value[level[0]], random.randint(-90, 90))
        kernel_second, anchor_second = genaratePsf(l_value[level[1]], random.randint(-90, 90))
        img_first = img.copy() if level[0]==0 else cv2.filter2D(img, -1, kernel_first, anchor=anchor_first)
        img_second = img if level[1]==0 else cv2.filter2D(img, -1, kernel_second, anchor=anchor_second)
    elif blur_type == 6:
        l_value = [0,20,15,10,5,0]
        img_first = img.copy() if level[0]==0 else F.image_compression(img, l_value[level[0]], '.jpg')
        img_second = img if level[1]==0 else F.image_compression(img, l_value[level[1]], '.jpg')
    elif blur_type == 7:
        l_value = [0,0.04,0.05,0.06,0.07,0.08]
        img_first = img.copy() if level[0]==0 else ghosting(img, l_value[level[0]])
        img_second = img if level[1]==0 else ghosting(img, l_value[level[1]])
    elif blur_type == 8:
        if random.random() < 0.3:
            l_value = [0,8,12,15,20,25]
            radial_type = 'angle'
        else:
            l_value = [0,5,7,9,11,12]
            radial_type = 'direction'
        if os.name == 'posix':
            h, w, c = img.shape
            img_first = img.copy()
            img_second = img
            if level[0] != 0:
                img_arr_f = img_first.ctypes.data_as(c_char_p)
                clib.apply_radial_blur(img_arr_f,h,w,l_value[level[0]],0 if radial_type=='angle' else 1)
            if level[1] != 0:
                img_arr_s = img_second.ctypes.data_as(c_char_p)
                clib.apply_radial_blur(img_arr_s,h,w,l_value[level[1]],0 if radial_type=='angle' else 1)
        else:
            img_first = img.copy() if level[0]==0 else apply_radial_blur(img.copy(), num=l_value[level[0]], type=radial_type)
            img_second = img if level[1]==0 else apply_radial_blur(img, num=l_value[level[1]], type=radial_type)
    else:
        l_value = [0,0.025,0.05,0.075,0.1,0.125]
        img_first = img.copy() if level[0]==0 else gasuss_noise(img, 0, l_value[level[0]])
        img_second = img if level[1]==0 else gasuss_noise(img, 0, l_value[level[1]])
    
    return img_first, img_second, 1 if level[0]>level[1] else -1


def get_motion_kernel(ksize):
    # ksize = random.choice(np.arange(self.motion_limit[0], self.motion_limit[1] + 1, 2))
    assert ksize > 2
    kernel = np.zeros((ksize, ksize), dtype=np.uint8)
    xs, ys = random.randint(0, ksize//5), random.randint(0, ksize - 1)
    # if random.random() < 0.5:
    #     xs = 0
    # else:
    #     ys = 0
    xe, ye = ksize-1 - xs, ksize-1 - ys
    
    cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)
    return kernel


class RandomErasing(DualTransform):
    """docstring for RandomErasing"""
    def __init__(self, always_apply=False, p=1.0, width=256, height=256,
                sl = 0.005, sh = 0.2, r1 = 0.1):
        super(RandomErasing, self).__init__(always_apply, p)
        self.width = width
        self.height = height
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.max_pixel = 255
        
    def apply(self, img, x, y, w, h, **params):
        select = random.choice([0,1,2,3])
        if select == 0:
            img[x:x+w, y:y+h, 0] = random.randint(0, self.max_pixel)
            img[x:x+w, y:y+h, 1] = random.randint(0, self.max_pixel)
            img[x:x+w, y:y+h, 2] = random.randint(0, self.max_pixel)
        elif select == 1:
            img[x:x+w, y:y+h, :] = np.random.rand(w, h, 3)*self.max_pixel
        elif select == 2:
            img[x:x+w, y:y+h, :] = np.array([0.4914, 0.4822, 0.4465])*random.randint(0, self.max_pixel)
        else:
            img[x:x+w, y:y+h, :] = np.array([0.4914, 0.4822, 0.4465])*self.max_pixel
        return img

    def apply_to_mask(self, mask, x, y, w, h, **params):
        mask[x:x+w, y:y+h] = 0
        return mask

    def get_params(self):
        area = self.height * self.width
        
        for attempt in range(100):
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                y = random.randint(0, self.height - h)
                x = random.randint(0, self.width - w)
                return {'x':x,'y':y,'w':w,'h':h}

        return {'x':0,'y':0,'w':1,'h':1}

if __name__=='__main__':
    # img = cv2.imread('F:/pingan/occ/tools/shiyuan.jpg')
    img = cv2.imread('/mnt/f/pingan/occ/tools/shiyuan.jpg')
    img = cv2.resize(img, (96, 96))
    # img_f, img_s, lbl = applyBlur(img)
    from ctypes import *
    clib = cdll.LoadLibrary('./radial_blur.so')
    img_arr = img.ctypes.data_as(c_char_p)
    # print(img)
    # print(img_arr)
    clib.apply_radial_blur(img_arr,96,96,50,0)

    cv2.imwrite('shiyuan_radial.jpg', img)
    # cv2.imshow('img1', img)
    # cv2.imshow('img2', img_s)
    # cv2.waitKey(0)
