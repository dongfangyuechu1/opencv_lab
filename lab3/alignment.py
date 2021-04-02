import math
import random

import cv2
import numpy as np

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)

    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        #BEGIN TODO 2
        #Fill in the matrix A in this loop.
        #Access elements using square brackets. e.g. A[0,0]
        #TODO-BLOCK-BEGIN
        n=2*i
        A[n]  = np.array([a_x,a_y,1,0,0,0,-b_x*a_x,-b_x*a_y,-b_x])
        A[n+1]= np.array([0,0,0,a_x,a_y,1,-b_y*a_x,-b_y*a_y,-b_y])
        #raise Exception("TODO in alignment.py not implemented")
        #TODO-BLOCK-END
        #END TODO

    U, s, Vt = np.linalg.svd(A)

    if A_out is not None:
        A_out[:] = A

    #s is a 1-D array of singular values sorted in descending order
    #U, Vt are unitary matrices
    #Rows of Vt are the eigenvectors of A^TA.
    #Columns of U are the eigenvectors of AA^T.

    #Homography to be calculated
    H = np.eye(3)

    #BEGIN TODO 3
    #Fill the homography H with the appropriate elements of the SVD
    #TODO-BLOCK-BEGIN
    Vt_row=Vt.shape[0]
    h = Vt[Vt_row-1]
    H[0]=h[0:3]
    H[1]=h[3:6]
    H[2]=h[6:9]
    #raise Exception("TODO in alignment.py not implemented")
    #TODO-BLOCK-END
    #END TODO

    return H

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''

    #BEGIN TODO 4
    #Write this entire method.  You need to handle two types of
    #motion models, pure translations (m == eTranslation) and
    #full homographies (m == eHomography).  However, you should
    #only have one outer loop to perform the RANSAC code, as
    #the use of RANSAC is almost identical for both cases.

    #Your homography handling code should call compute_homography.
    #This function should also call get_inliers and, at the end,
    #least_squares_fit.
    #TODO-BLOCK-BEGIN
    max_num = -1
    
    if m == eTranslate:
        for i in range(nRANSAC):
            random_index = np.random.randint(0,len(matches), size=1)
            random_match = matches[random_index[0]]
            (x0, y0) = f1[random_match.queryIdx].pt
            (x, y) = f2[random_match.trainIdx].pt
            H = np.array([[1, 0, x - x0], [0, 1, y - y0], [0, 0, 1]])
            inliers = getInliers(f1, f2, matches, H, RANSACthresh)
            inliers_length=len(inliers)
            if  max_num <  inliers_length:
                max_num=inliers_length
                max_inliers=inliers
        M = leastSquaresFit(f1, f2, matches, eTranslate,  max_inliers)

    elif m == eHomography:
        for i in range(nRANSAC):
            #rand set
            random_index = np.random.randint(0, len(matches), size=4)
            random_matches=[matches[random_index[0]],matches[random_index[1]],matches[random_index[2]],matches[random_index[3]]]
            H = computeHomography(f1,f2, random_matches)
            inliers = getInliers(f1, f2, matches, H, RANSACthresh)
            inliers_length=len(inliers)
            if  max_num <  inliers_length:
                max_num=inliers_length
                max_inliers=inliers
        M = leastSquaresFit(f1, f2, matches, eHomography, max_inliers)
    #raise Exception("TODO in alignment.py not implemented")
    #TODO-BLOCK-END
    #END TODO
    return M

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''
    inlier_indices = []

    for i in range(len(matches)):
        #BEGIN TODO 5
        #Determine if the ith matched feature f1[id1], when transformed
        #by M, is within RANSACthresh of its match in f2.
        #If so, append i to inliers
        #TODO-BLOCK-BEGIN
        i_match = matches[i]
        (x0, y0) = f1[i_match.queryIdx].pt
        (x, y) = f2[i_match.trainIdx].pt
        f1_=np.array([[x0,y0,1]])
        f2_=np.array([[x,y,1]])
        f1_alt = np.dot(M, f1_.T).T
        f1_alt =f1_alt /f1_alt[0][2]
        euc_distance = np.linalg.norm(f2_ - f1_alt)
        if euc_distance<RANSACthresh:
            inlier_indices.append(i)
        #raise Exception("TODO in alignment.py not implemented")
        #TODO-BLOCK-END
        #END TODO

    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate)
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == eTranslate:
        #For spherically warped images, the transformation is a
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        for i in range(len(inlier_indices)):
            #BEGIN TODO 6
            #Use this loop to compute the average translation vector
            #over all inliers.
            #TODO-BLOCK-BEGIN
            k = matches[inlier_indices[i]]
            (x0, y0) = f1[k.queryIdx].pt
            (x, y) = f2[k.trainIdx].pt
            x_ = x - x0
            y_ = y - y0
            u += x_
            v += y_
            #raise Exception("TODO in alignment.py not implemented")
            #TODO-BLOCK-END
            #END TODO

        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0,2] = u
        M[1,2] = v

    elif m == eHomography:
        #BEGIN TODO 7
        #Compute a homography M using all inliers.
        #This should call computeHomography.
        #TODO-BLOCK-BEGIN
        new_match = []
        inlier_indices_length=len(inlier_indices)
        for i in range(inlier_indices_length):
            index = inlier_indices[i]
            new_match.append(matches[index])            
        M = computeHomography(f1, f2, new_match)
        #raise Exception("TODO in alignment.py not implemented")
        #TODO-BLOCK-END
        #END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M

