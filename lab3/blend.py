import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         maxX: int for the maximum X value of a corner
         maxY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN
    img_row, img_col = img.shape[:2]
    cor = np.array([[0, 0, img_col - 1, img_col - 1],[0, img_row - 1, 0,img_row - 1],[1, 1, 1, 1]])
    res = np.dot(M, cor)
    res = res / res[-1]
    minX = np.min(res, axis=1)[0]
    minY = np.min(res, axis=1)[1]
    maxX = np.max(res, axis=1)[0]
    maxY = np.max(res, axis=1)[1]
    #raise Exception("TODO in blend.py not implemented")
    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN
    acc_length, acc_width = acc.shape[:2]
    img_length, img_width = img.shape[:2]
    img = cv2.copyMakeBorder(img, 0, acc_length - img_length, 0,  acc_width - img_width, cv2.BORDER_CONSTANT, value=0)
    img_length, img_width = img.shape[:2]
    x_range = np.arange(img_width)
    y_range = np.arange(img_length)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    matrix_ones = np.ones((img_length, img_width))
    coor = np.dstack((x_grid, y_grid,matrix_ones))
    coor =coor.reshape((img_length * img_width, 3)).T
    loc = np.linalg.inv(M).dot(coor)
    loc = loc / loc[2]
    
    map1 = loc[0].reshape((img_length, img_width)).astype(np.float32)
    map2 = loc[1].reshape((img_length, img_width)).astype(np.float32)
    img_map = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    minX, minY, maxX, maxY = imageBoundingBox(img, M)
    img_map_row=img_map.shape[0]
    img_map_col=img_map.shape[1]
    dst_one = np.ones((img_map_row, img_map_col, 1))
    dst_img = np.dstack((img_map, dst_one))

    #fethered
    k = 1 / blendWidth
    lower=-k * minX
    upper= k * (acc_width - 1 - minX)
    right = np.linspace(lower,upper, acc_width)
    for i in range(len(right)):
        if right[i] > 1:
            right[i] = 1
        elif right[i] < 0:
            right[i] = 0
    right =right.reshape((1, acc_width, 1))
    l_ones=np.ones((1, acc_width, 1))
    left = l_ones - right
    img_feathered = right * dst_img
    acc *= left

    co_acc = acc[:, :, 0:3].astype(np.uint8)
    g_acc = cv2.cvtColor(co_acc, cv2.COLOR_BGR2GRAY)
    g_img = cv2.cvtColor(img_map, cv2.COLOR_BGR2GRAY)
    mask_img =np.array(g_img).astype(bool)
    mask_img =mask_img.reshape((acc_length, acc_width, 1))
    mask_acc =np.array(g_acc).astype(bool)
    mask_acc =mask_acc.reshape((acc_length, acc_width, 1))

    img_masked = mask_img * img_feathered
    acc *= mask_acc
    acc += img_masked
    #raise Exception("TODO in blend.py not implemented")
    #TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    acc_row = acc.shape[0]
    acc_col = acc.shape[1]    
    for i in range(acc_row):
        for j in range(acc_col):
            if acc[i][j][3] == 0:
                acc[i][j][3] = 1
    acc_n0f= acc[:, :, 3].reshape((acc_row,acc_col, 1))
    img = acc / acc_n0f
    img = img[:, :, 0:3]
    img = img.astype(np.uint8)
    #raise Exception("TODO in blend.py not implemented")
    #TODO-BLOCK-END
    # END TODO
    return img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accHeight: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = sys.maxsize
    minY = sys.maxsize
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        m = imageBoundingBox(img, M)
        # print(mset)
        minX = min(minX, m[0])
        minY = min(minY, m[1])
        maxX = max(maxX, m[2])
        maxY = max(maxY, m[3])
        #raise Exception("TODO in blend.py not implemented")
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    #TODO-BLOCK-BEGIN
    if is360==True:
        A = computeDrift(x_init, y_init, x_final, y_final, width)
    #raise Exception("TODO in blend.py not implemented")
    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

