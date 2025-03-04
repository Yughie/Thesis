import numpy as np
import cv2 as cv
import glob
import pickle
import os

# Chessboard settings
chessboardSize = (9, 6)
frameSize = (1280, 720)
square_size = 22  # Size of a square in mm

# Termination Criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), ... (6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
objp *= square_size  # Scale by square size

# Storage for points
objpoints = []  # 3D points in real world
imgpointsL = []  # 2D points for left camera
imgpointsR = []  # 2D points for right camera

# Load images
imagesLeft = glob.glob('left_camera_frame/*.png')
imagesRight = glob.glob('right_camera_frame/*.png')

# Detect chessboard corners
for imgLeft, imgRight in zip(imagesLeft, imagesRight):
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find chessboard corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    if retL and retR:
        objpoints.append(objp)
        
        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)

        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

        # Draw detected corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('Left Chessboard', imgL)
        cv.imshow('Right Chessboard', imgR)
        cv.waitKey(500)

cv.destroyAllWindows()

# Check if calibration file exists
calib_file = "stereo_calibration_data.pkl"
if os.path.exists(calib_file):
    print("🔹 Loading saved calibration data...")
    with open(calib_file, "rb") as f:
        calibration_data = pickle.load(f)

    cameraMatrixL = calibration_data["cameraMatrixL"]
    distL = calibration_data["distL"]
    cameraMatrixR = calibration_data["cameraMatrixR"]
    distR = calibration_data["distR"]
    rot = calibration_data["rot"]
    trans = calibration_data["trans"]
    essentialMatrix = calibration_data["essentialMatrix"]
    fundamentalMatrix = calibration_data["fundamentalMatrix"]

else:
    print("🔹 Performing new calibration...")

    # Individual Camera Calibration
    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
    newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, frameSize, 1, frameSize)

    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
    newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, frameSize, 1, frameSize)

    # Stereo Calibration
    flags = cv.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    retStereo, cameraMatrixL, distL, cameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR, cameraMatrixL, distL, cameraMatrixR, distR, frameSize,
        criteria=criteria_stereo, flags=flags
    )

    # Save calibration results
    calibration_data = {
        "cameraMatrixL": cameraMatrixL,
        "distL": distL,
        "cameraMatrixR": cameraMatrixR,
        "distR": distR,
        "rot": rot,
        "trans": trans,
        "essentialMatrix": essentialMatrix,
        "fundamentalMatrix": fundamentalMatrix
    }

    with open(calib_file, "wb") as f:
        pickle.dump(calibration_data, f)
    print("✅ Calibration data saved!")

# Stereo Rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(cameraMatrixL, distL, cameraMatrixR, distR, frameSize, rot, trans)

# Undistort and rectify images
left_map1, left_map2 = cv.initUndistortRectifyMap(cameraMatrixL, distL, R1, P1, frameSize, cv.CV_16SC2)
right_map1, right_map2 = cv.initUndistortRectifyMap(cameraMatrixR, distR, R2, P2, frameSize, cv.CV_16SC2)

# Process images for verification
for imgLeft, imgRight in zip(imagesLeft, imagesRight):
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)

    # Apply rectification maps
    rectifiedL = cv.remap(imgL, left_map1, left_map2, cv.INTER_LINEAR)
    rectifiedR = cv.remap(imgR, right_map1, right_map2, cv.INTER_LINEAR)

    # Save rectified images
    cv.imwrite(f"rectified_left/{os.path.basename(imgLeft)}", rectifiedL)
    cv.imwrite(f"rectified_right/{os.path.basename(imgRight)}", rectifiedR)

    # Show rectified images
    rectified_pair = np.hstack((rectifiedL, rectifiedR))
    cv.imshow("Rectified Stereo Pair", rectified_pair)
    cv.waitKey(500)

cv.destroyAllWindows()
print("✅ Rectified images saved and displayed!")
