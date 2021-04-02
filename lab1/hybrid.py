import sys
sys.path.append('/Users/bin/opencv-3.1.0ildb/')
from PIL import Image
import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    img_array=np.array(img)
    img_v = img_array.ndim    
    img_row = img_array.shape[0]      
    img_column = img_array.shape[1]  
    kernel_row = kernel.shape[0]            
    kernel_column = kernel.shape[1]
    pad_lr=int(kernel_row/2)
    pad_ud=int(kernel_column/2)
    if img_v==2:
        cor_array = np.zeros((img_row,img_column))
        pad=np.pad(img_array,((pad_lr,pad_lr),(pad_ud,pad_ud)),'constant',constant_values=(0,0))
        for j in range(0,img_row):
            for k in range(0,img_column):
                    cor_array[j][k] = (pad[j:j+kernel_row,k:k+kernel_column]*kernel).sum()
                    if cor_array[j][k]>255:
                        cor_array[j][k]=255
                    elif cor_array[j][k]<0:
                        cor_array[j][k]=0
        return cor_array
    if img_v==3:
        img_c=img_array.shape[2]
        cor_array = np.zeros((img_row,img_column,img_c))
        for i in range(0,img_c):
            pad=np.pad(img_array[:,:,i],((pad_lr,pad_lr),(pad_ud,pad_ud)),'constant',constant_values=(0,0))
            for j in range(0,img_row):
                for k in range(0,img_column):
                    cor_array[j][k][i]= (pad[j:j+kernel_row,k:k+kernel_column]*kernel).sum()
                    if cor_array[j][k][i]>255:
                        cor_array[j][k][i]=255
                    elif cor_array[j][k][i]<0:
                        cor_array[j][k][i]=0
        return cor_array
    # TODO-BLOCK-BEGIN
    #raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    rev_kernel = np.flipud(np.fliplr(kernel))
    con_array=cross_correlation_2d(img,rev_kernel)
    return con_array

    # TODO-BLOCK-BEGIN
    #raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
    c_row = height//2
    c_column = width//2
    k = 2*(sigma**2)
    gaussian_kernel = np.zeros((height,width))
    sum=0
    for i in range(height):
        for j in range(width):
            x = i - c_row
            y = j - c_column
            coefficient=1.0/(np.pi*k)
            index=-(x**2+y**2)/k
            gaussian_kernel[i][j]=coefficient*np.exp(index)
            sum+=gaussian_kernel[i][j]
    gaussian_kernel/=sum  
    return gaussian_kernel

    # TODO-BLOCK-BEGIN
    #raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    low_kernel=gaussian_blur_kernel_2d(sigma,size,size)
    return convolve_2d(img, low_kernel)

    # TODO-BLOCK-BEGIN
    #raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def high_pass(img, sigma, size):   
    img_array=np.array(img)
    low_array=low_pass(img,sigma,size)
    return img_array-low_array

    # TODO-BLOCK-BEGIN
    #raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
    high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
   #return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
    min_h=np.amin(hybrid_img);
    max_h=np.amax(hybrid_img);
    return ((hybrid_img-min_h)/(max_h-min_h) * 255).clip(0, 255).astype(np.uint8)
