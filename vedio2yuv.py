import cv2
import numpy as np

FPS = 15 
Width = 320
Height = 240

class pixel2yuv:
    '''
    CONVERT a single frame to YUV
    '''
    def __init__(self, width, height,frame, buffer, idx_frames):
        self.frame = frame
        self.buffer = buffer
        self.idx_frames = idx_frames
        self.height = height
        self.width =  width

    def get_data_framesize(self):
        return (self.width * self.height * 3 / 2) # 每4个像素共享一个UV，所以申请len*3/2个YUV长度；其中Y占len个，U和V分别占用len/4个

    def get_yuv_index(self, width=None, height=None):
        if not width:
            wh = self.width * self.height
        else:
            wh = width * height

        idx = self.idx_frames * 1.5
        Y_idx = int(idx * wh)
        U_idx = Y_idx + int(wh/4)
        V_idx = U_idx + int(wh/4)
        return (Y_idx, U_idx, V_idx)#计算Y、U、V在数组中的index
        
    def trans(self):
        y_idx = self.get_yuv_index()[0]
        u_idx = self.get_yuv_index()[1]
        v_idx = self.get_yuv_index()[2]
        for h in range(0, self.height):             
            for w in range(0, self.width):
                self.buffer[y_idx] = 0.299 * self.frame[h, w, 2] + 0.587 * self.frame[h, w, 1] + 0.114 * self.frame[h, w, 0]
                y_idx += 1

                if h % 2 == 0:
                    if w % 2 == 0:
                        self.buffer[u_idx] = -0.16 * self.frame[h, w, 2] + -0.33*self.frame[h, w, 1] + 0.5 * self.frame[h, w, 0]
                        self.buffer[v_idx] = 0.5 * self.frame[h, w, 2] + -0.419 * self.frame[h, w, 1] + -0.00813* self.frame[h, w ,0]

                        u_idx += 1
                        v_idx += 1
        
        self.buffer = np.clip(self.buffer, 0, 255)

        return self.buffer
    


def video2yuv(vid_path, yuv='420'):
    video = vid_path
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError('cap is not opened !')
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < FPS:
            raise ValueError('fps of video is lower than 15 !')

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print('orgin_frame{}'.format(frame_num))
        print("current video is {:.0f}*{:.0f}, {:.0f} fps, frame_num = {:.0f}\n".format(width, height, fps, frame_num))

        # down sample the video to 15 fps
        sample_step = int(fps/FPS)+1
        aim_frame_num = int(frame_num * float(FPS)/fps)
        print('aim_frame{}'.format(aim_frame_num))
        srcframe_idx = 0  # the src video frame index
        dstframe_idx = 0  # the dst (15 fps) video frame index
        buffer_bytes = np.zeros(int(Width * Height * 1.5 * aim_frame_num), dtype='int32')

        retval, frame = cap.read()
        while retval:

            # down sample
            if srcframe_idx % sample_step == 0:
                print('processing {}th frame'.format(dstframe_idx))

                # Attention, resize function scalar factor is (width, height), not as a frame index (height, width)
                resize_frame = cv2.resize(frame, (Width, Height), interpolation=cv2.INTER_CUBIC)
                pix_frame = pixel2yuv(Width,Height,resize_frame, buffer_bytes, dstframe_idx)
                buffer_bytes = pix_frame.trans()
                srcframe_idx += 1
                dstframe_idx += 1
                retval, frame = cap.read()
            else:
                retval, frame = cap.read()
                srcframe_idx += 1

        cap.release()
        return buffer_bytes.astype(dtype='uint8').tobytes()


# if __name__ == 'main':
buffer = video2yuv('C:/Users/Administrator/Desktop/test.mp4')
with open('output2.yuv', 'wb') as f:
    f.write(buffer)
    f.close()


