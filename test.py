import numpy as np

pattern_size = (3, 2) #(9, 6) # column 수, row 수
chessboard_size = (3, 2) #(9, 6)
square_size = 0.034
# the real-world dimension of a chessboard square, in meters
# Real-world 3D corner "positions"
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
print("pattern_points:", pattern_points.shape)
print("pattern_points:", pattern_points)
# https://github.com/opencv/opencv/issues/9150#issuecomment-674664643
pattern_points = np.expand_dims(np.asarray(pattern_points), -2)
print("pattern_points:", pattern_points.shape)
print("pattern_points:", pattern_points)
pattern_points *= square_size


#####

objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
"""
- np.mgrid: 
  - (2, column 수, row 수): (column 수, row 수)행렬이 2개인데, 
  - 각각 height 좌표, width 좌표를 의미
- (2,  column 수, row 수).T = (row 수, column 수, 2)
- reshape: (column 수 * row 수, 2)
- 이 뜻은, 
  - ( column 수, row 수) array를, 
  - height 순으로 읽어내려가겟다는 뜻이다. (세로 줄 하나씩 위에서 아래로 읽어가겠다는 뜻)
- objp[:, 0]: height index
- objp[:, 1]: width index
"""
objp[:, :2] = np.mgrid[0:chessboard_size[0],
              0:chessboard_size[1]].T.reshape(-1, 2)
print("objp:", objp.shape)
print("objp:", objp)