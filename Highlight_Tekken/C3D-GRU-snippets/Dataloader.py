import os
import cv2
import numpy as np

# get Training Highlight videos
def get_videos(dataroot, is_train, is_highlight):
    videofiles = os.listdir(dataroot)
    videofiles = [os.path.join(dataroot, v) for v in videofiles]

    n_videos = len(videofiles)

    total_out = []
    total_label = []

    for idx in range(n_videos): # for one video
        cap = cv2.VideoCapture(videofiles[idx])

        frames = []

        while True:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()

        out = np.concatenate(frames)
        out = out.reshape(-1, 270, 480, 3)

        if is_train: # Train set
            if is_highlight:
                label = np.ones(out.shape[0])
            else:
                label = np.zeros(out.shape[0])
        else: # Test set
            filename = os.path.split(videofiles[idx])[-1]
            h_start = filename.index("(")
            h_end = filename.index(")")
            h_frames = filename[h_start + 1: h_end]

            label = np.zeros(out.shape[0])

            if "," in h_frames:
                s, e = h_frames.split(',')
                label[int(s) : int(e)] = 1.

        total_out.append(out)
        total_label.append(label)

    return total_out, total_label

def plotVideo(frames):
    for f in frames:
        cv2.imshow("frame", f)
        cv2.waitKey(10)

if __name__ == "__main__":
    hv, hv_label = get_videos("C:\\Users\young\Downloads\PROGRAPHY DATA_ver3\HV",
                              is_train=True,
                              is_highlight=True)
    rv, rv_label = get_videos("C:\\Users\young\Downloads\PROGRAPHY DATA_ver3\RV",
                              is_train=True,
                              is_highlight=False)
    tv, tv_label = get_videos("C:\\Users\young\Downloads\PROGRAPHY DATA_ver3\\testRV",
                                is_train=False,
                                is_highlight=None)

    for frames, label in zip(hv, hv_label):
        print(frames.shape)
        print(label.shape)

    for frames, label in zip(rv, rv_label):
        print(frames.shape)
        print(label.shape)

    for frames, label in zip(tv, tv_label):
        print(frames.shape)
        print(label.shape)

