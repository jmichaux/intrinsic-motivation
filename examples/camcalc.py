import numpy as np

def cam_pose(pos, lookat):
    z_hat = np.array([0,0,-1])
    pos = np.array(pos)
    lookat = np.array(lookat)

    z = pos - lookat
    x = np.cross(z, z_hat)
    y = np.cross(z, x)

    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)

    xy = list(x.round(decimals=2)) + list(y.round(decimals=2))
    xy = str(xy).replace(',', '')
    xy = str(xy).replace('[', '')
    xy = str(xy).replace(']', '')
    return xy

if __name__ == '__main__':
    pos1 = [1.2, 1.2, 1.2]
    pos2 = [1.2, -1.2, 1.2]
    lookat = [0.65, 0.005, 0.535]
    print(cam_pose(pos1, lookat))
    print(cam_pose(pos2, lookat))
