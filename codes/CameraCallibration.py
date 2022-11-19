import numpy as np
import cv2 as cv
import glob

# name of the file to be calibrated:
target_file = 'target1.jpg'
result_name = 'calibrated_result_' + target_file


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0, 0, 0), (1, 0, 0), ... (9, 7, 0)
# we have a 9x7 grid
objp = np.zeros((8*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

# arrays to store object points and image points from all the images
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

# searches through directories to find all training images
images = glob.glob('./codes/training_images/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (8, 6), None)
    
    # if found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        cv.drawChessboardCorners(img, (8, 6), corners2, ret)
        
        # Resize the image to be shown:
        cv.namedWindow("Resized_Window", cv.WINDOW_NORMAL)
        cv.resizeWindow("Resized_Window", 2000, 4000)
        cv.imshow('Resized_Window', img)
        # press q to iterate to the next images 
        if cv.waitKey(3000) == ord('q'):
            continue

cv.destroyAllWindows()

# select the image that needs to be calibrated based on training data
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
img = cv.imread('./codes/target_images/' + target_file)
h, w = img.shape[:2]
new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv.undistort(img, mtx, dist, None, new_camera_matrix)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('./codes/results/' + result_name, dst)