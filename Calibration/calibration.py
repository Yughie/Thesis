import cv2
import numpy as np
import glob  # To easily find checkerboard images

# --- Configuration ---
CHECKERBOARD_SIZE = (9, 6)  # (Columns, Rows) - Number of INNER corners in the checkerboard pattern
SQUARE_SIZE_MM = 25.0  # Size of each square in millimeters (MEASURE THIS ACCURATELY!)

# --- Create object points (3D points in real world space) ---
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_MM  # Scale by the square size to get real-world units (mm)

# This 'objp' is the same for all calibration images because the checkerboard is rigid and of fixed size.
# We'll reuse this for all images.


# --- Lists to store object points and image points for ALL calibration images ---
objpoints = []  # 3d point in real world space
imgpoints_left = []  # 2d points in image plane for left camera.
#imgpoints_right = [] # 2d points in image plane for right camera.

# --- Calibration for LEFT camera ---
print("Calibrating LEFT camera...")
images_left = glob.glob('./calibration_images/left_cam/*.jpg') # Path to your left camera images
for fname in images_left:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp) # Append the SAME objp for each image

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints_left.append(corners2)

        # Draw and display the corners (for visualization - optional)
        cv2.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners2, ret)
        cv2.imshow('Left Camera Corner Detection', img)
        cv2.waitKey(1) # Wait briefly to show image

cv2.destroyAllWindows()


# --- Calibration for RIGHT camera --- (Similar process, adjust paths and variable names)
"""
print("Calibrating RIGHT camera...")
images_right = glob.glob('./calibration_images/right_cam/*.jpg') # Path to your right camera images
imgpoints_right = [] # Clear imgpoints_right list for right camera
for fname in images_right:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints_right.append(corners2)

cv2.destroyAllWindows()
"""

# --- Calibration for STEREO pairs --- (Similar, but process pairs of images)
"""
print("Preparing stereo calibration data...")
images_stereo_left = sorted(glob.glob('./calibration_images/stereo_cam/*_left.jpg')) # Sorted to ensure pairs match
images_stereo_right = sorted(glob.glob('./calibration_images/stereo_cam/*_right.jpg'))

stereo_imgpoints_left = []
stereo_imgpoints_right = []

for i in range(len(images_stereo_left)): # Loop through stereo pairs
    img_left = cv2.imread(images_stereo_left[i])
    img_right = cv2.imread(images_stereo_right[i])
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD_SIZE, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD_SIZE, None)

    if ret_left == True and ret_right == True: # Ensure checkerboard found in BOTH images of the pair
        stereo_imgpoints_left.append(cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1),
                                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)))
        stereo_imgpoints_right.append(cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1),
                                                      (cv2.TERM_CRITERIA_EPS + cv2.
                                                      TERM_CRITERIA_MAX_ITER, 30, 0.001)))
"""
        

# --- Calibrate LEFT camera ---
print("Performing LEFT camera calibration...")
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    objpoints, imgpoints_left, gray.shape[::-1], None, None) # gray.shape[::-1] gets image size (width, height)

print("LEFT Camera Calibration RMS error:", ret_left)
print("LEFT Camera Matrix (mtx_left):\n", mtx_left)
print("LEFT Distortion Coefficients (dist_left):\n", dist_left)

"""
# --- Calibrate RIGHT camera --- (Similar process)
print("Performing RIGHT camera calibration...")
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    objpoints, imgpoints_right, gray.shape[::-1], None, None)

print("RIGHT Camera Calibration RMS error:", ret_right)
print("RIGHT Camera Matrix (mtx_right):\n", mtx_right)
print("RIGHT Distortion Coefficients (dist_right):\n", dist_right)
"""


### ADDITIONAL FOR LEFT CALIRATION ONLY
# --- Save Calibration Data (LEFT camera only) ---
print("Saving LEFT camera calibration data...")
calibration_data_path = "./left_camera_calibration.npz" # Specific filename for left camera

np.savez(calibration_data_path,
         cameraMatrix_left=mtx_left, distCoeffs_left=dist_left) # Saved with "_left" suffix

print("LEFT camera calibration data saved to", calibration_data_path)

print("LEFT Camera Calibration process complete.")

"""
# --- Stereo Calibration ---
print("Performing stereo calibration...")
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC # Fix intrinsic parameters from individual calibrations
#flags |= cv2.CALIB_RATIONAL_MODEL # Optional: Try rational model if distortion is high
#flags |= cv2.CALIB_SAME_FOCAL_LENGTH # Optional: If you expect same focal length

stereocalib_retval, stereo_cameraMatrix1, stereo_distCoeffs1, stereo_cameraMatrix2, stereo_distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, stereo_imgpoints_left, stereo_imgpoints_right,
    mtx_left, dist_left, mtx_right, dist_right,
    gray_left.shape[::-1], None, None, None, None, flags=flags) # Use gray_left.shape for image size

print("Stereo Calibration RMS error:", stereocalib_retval)
print("Rotation matrix (R):\n", R)
print("Translation vector (T):\n", T)

"""