import cv2
import numpy as np
import glob
import os

# 체스보드 패턴 생성
# 체스보드 크기
"""
the number of *inner* points!
So for a grid of 10x7 *squares* there's 9x6 inner points
"""
chessboard_size = (9, 6)  # column 수, row 수
# square_size: the real-world dimension of a chessboard square, in meters
square_size = 0.034
# https://velog.io/@hsbc/아이폰-카메라의-이미지-센서-픽셀-크기
sensor_size = (4.00, 3.00) # (mm)
# pattern_points: (9 * 6, 3)
pattern_points = np.zeros((np.prod(chessboard_size), 3), np.float32)
"""
- np.mgrid: 
  - (2, column 수, row 수): (column 수, row 수)행렬이 2개인데, 
  - 각각 column 좌표, width 좌표를 의미
- (2,  column 수, row 수).T = (row 수, column 수, 2)
- reshape: (column 수 * row 수, 2)
- 이 뜻은, 
  - ( column 수, row 수) array를, 
    - 가로 줄 하나씩 왼쪽에서 오른쪽으로 읽어가겠다는 뜻
- pattern_points[:, 0]: height index
- pattern_points[:, 1]: width index
"""
pattern_points[:, :2] = np.mgrid[0:chessboard_size[0],
                                 0:chessboard_size[1]].T.reshape(-1, 2)
# https://github.com/opencv/opencv/issues/9150#issuecomment-674664643
pattern_points = np.expand_dims(np.asarray(pattern_points), -2)  # (9*6, 3, 1)
pattern_points *= square_size

# 30: TERM_CRITERIA_COUNT -> 알고리즘이 멈추는 최대 반복 횟수
# 0.001: TERM_CRITERIA_EPS -> 알고리즘이 수렴하는 허용 오차
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 객체 포인트와 이미지 포인트 저장할 배열
obj_points = []
img_points = []
# 이미지 로드
dir = "input_jpg"
images = glob.glob(f"{dir}/*.jpg")
corner_dir = "output"
if os.path.exists(corner_dir):
    os.system(f"rm -rf {corner_dir}")
os.makedirs(corner_dir)
height_resolution = None
width_resolution = None
image_name_to_image = {}
for image in images:
    print("image: ", image)
    img = cv2.imread(image)
    image_name_to_image[image] = img
    if height_resolution is None or width_resolution is None:
        height_resolution, width_resolution = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # corners: np.array (N, 1, 2)
    ret, corners = cv2.findChessboardCorners(image=gray,
                                             patternSize=chessboard_size,
                                             corners=None)
    print("ret: ", ret)
    if ret:
        obj_points.append(pattern_points)
        # corners2: cv2.typing.MatLike # cv2.mat_wrapper.Mat, NumPyArrayNumeric
        # corners2: np.array (N, 1, 2)
        corners2 = cv2.cornerSubPix(image=gray,
                                    corners=corners,
                                    winSize=(11, 11),
                                    zeroZone=(-1, -1),
                                    criteria=criteria)
        if isinstance(corners2, str):
            raise ValueError(f"Failed to find corners for {image}")
        img_points.append(corners2)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # 코너를 이미지에 그립니다.
        cv2.drawChessboardCorners(vis,
                                  patternSize=chessboard_size,
                                  corners=corners2,
                                  patternWasFound=ret)

        # 결과 이미지를 저장합니다.
        image_file_name = image.split("/")[-1]
        output_image_path = os.path.join(corner_dir, f"{image_file_name}")
        cv2.imwrite(output_image_path, vis)
        print(f"Saved {output_image_path}")
(rms_error, camera_matrix, dist_coefs, rvecs,
 tvecs) = cv2.calibrateCamera(obj_points,
                              img_points,
                              imageSize=(width_resolution, height_resolution),
                              cameraMatrix=None,
                              distCoeffs=None)
"""
rms_error: float
    - 전체 Root Mean Square re-projection error
    - 이 값은 캘리브레이션 과정의 품질을 나타내며, 낮을수록 더 정확한 캘리브레이션 결과를 의미
    - 0.4043353
camera_matrix: np.ndarray (3, 3)
    - intrinsic matrix
    - [[2.82941040e+03 0.00000000e+00 1.97697417e+03]
 [0.00000000e+00 2.82939480e+03 1.46900315e+03]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

dist_coefs: np.ndarray (1, 5)
    - `[k1, k2, p1, p2, k3]`을 포함하며, 
    - 이는 각각 반경 왜곡 계수(k1, k2, k3)와 접선 왜곡 계수(p1, p2)
    [[ 0.09491546 -0.22681414  0.00043416  0.00034345  0.18057507]]

rvecs: Tuple[np.ndarray]
    - (np.ndarray (3, 1), ...)
    - 각 패턴 뷰에 대한 회전 벡터의 튜플.
    - 각 뷰에 대한 3x1 회전 벡터로, 월드 좌표계를 카메라 좌표계로 변환하는 회전을 나타냅니다.
    - 이 벡터는 Rodrigues 변환을 통해 회전 행렬로 변환할 수 있습니다.
    -  (array([[ 0.01413114],
       [ 0.01593625],
       [-0.00571477]]),)

tvecs: Tuple[np.ndarray]
    - 설명: 각 패턴 뷰에 대한 변환 벡터의 튜플.
    - 값의 의미:
      - 각 뷰에 대한 3x1 변환 벡터로, 월드 좌표계를 카메라 좌표계로 변환하는 평행 이동을 나타냄
      - `(array([[-3.89025738],
        [-2.26881858],
        [ 7.43518093]]),)`

"""
(newcameramtx, roi) = cv2.getOptimalNewCameraMatrix(
    cameraMatrix=camera_matrix,
    distCoeffs=dist_coefs,
    imageSize=(width_resolution, height_resolution),
    alpha=1,
    newImgSize=(width_resolution, height_resolution))
"""
newcameramtx: np.ndarray (3, 3)
    - `출력되는 새로운 카메라 intrinsic 행렬.`
    - [[2.86704074e+03 0.00000000e+00 1.97858396e+03]
 [0.00000000e+00 2.86554634e+03 1.47026838e+03]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

roi: Tuple[int]
    - 선택적 출력 사각형으로, `왜곡 보정된 이미지에서 모든 유효 픽셀 영역을 둘러싼 사각형`
    - (x, y, width_resolution, height_resolution)
    - ROI 영역
    - (14, 12, 3997, 2995)
"""
# dst: 보정된 이미지
for image_name, img_ in image_name_to_image.items():
    dst = cv2.undistort(src=img_,
                        cameraMatrix=camera_matrix,
                        distCoeffs=dist_coefs,
                        dst=None,
                        newCameraMatrix=newcameramtx)
    x, y, width_resolution, height_resolution = roi
    dst = dst[y:y + height_resolution, x:x + width_resolution]
    image_file_name = image_name.split("/")[-1].split(".")[0]
    cv2.imwrite(f'calibresult_{image_file_name}.png', dst)

# 에러 계산
tot_error = 0
for camera_idx in range(len(obj_points)):
    # objpoint: 실제 세계의 3D 점 (H*W, 3)
    # rvec: 각 패턴 뷰에 대한 회전 벡터의 튜플.
    # tvec: 각 패턴 뷰에 대한 변환 벡터의 튜플.
    # camera_matrix: intrinsic matrix (3, 3) # optimal 은 아님
    # dist_coefs: distortion coefficients (1, 5)
    # object point를 이미지 point로 변환
    img_points2, _ = cv2.projectPoints(obj_points[camera_idx],
                                       rvecs[camera_idx], tvecs[camera_idx],
                                       camera_matrix, dist_coefs)
    # cv2.findChessboardCorners + cv2.cornerSubPix로 얻은 이밎 point와,
    # 변환된 이미지 point와 거리 계산
    error = cv2.norm(img_points[camera_idx], img_points2,
                     cv2.NORM_L2) / len(img_points2)
    tot_error += error
print("total error: ", tot_error / len(obj_points))

if sensor_size is not None:
    assert isinstance(sensor_size, tuple) and len(sensor_size) == 2
    (fovx, fovy, focal_length,
     principal_point, aspect_ratio) = cv2.calibrationMatrixValues(
         camera_matrix, (width_resolution, height_resolution), sensor_size[0],
         sensor_size[1])
    print("fovx: ", fovx)
    print("fovy: ", fovy)
    print("focal_length: ", focal_length)
    print("principal_point: ", principal_point)
    print("aspect_ratio: ", aspect_ratio)

cv2.destroyAllWindows()
