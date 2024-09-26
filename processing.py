#! /home/rennerph/.conda/envs/EVM/bin/python


import numpy as np
import cv2
import scipy.fftpack as fftpack
from scipy import signal
import platform


class VideoHandling():
    #Lab colour space recherchieren
    def YIQ2RGB(self, video):
        rgb_from_yiq = np.array([[1, 0.956, 0.621],
                             [1, -0.272, -0.647],
                             [1, -1.106, 1.703]])
        t = np.dot(video, rgb_from_yiq.T)
        return t
    
    def RGB2YIQ(self, video):
        yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.274, -0.322],
                             [0.211, -0.523, 0.312]])
        t = np.dot(video, yiq_from_rgb.T)
        return t
    
    def RGB2LAB(self,video):
        for i in range(0, video.shape[0]):
            video[i] = cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR)
            video[i] = cv2.cvtColor(video[i], cv2.COLOR_BGR2Lab)
        return video
    
    def LAB2RGB(self,video):
        for i in range(0, video.shape[0]):
            video[i] = cv2.cvtColor(video[i], cv2.COLOR_Lab2BGR)
            video[i] = cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)
        return video
    
    def SaveVideo(self, video, fps, name):
        print('Save video')

        if platform.system() == 'Linux':
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        [height, width] = video[0].shape[0:2]
        writer = cv2.VideoWriter(name[:-4] + "OUTPUT.avi", fourcc, fps, (width, height), True)
        for i in range(video.shape[0]):
            writer.write(cv2.convertScaleAbs(video[i]))
        writer.release()

    def ResizeVideo(self, video, num_frames,  width, height):
        resized_video = np.zeros((num_frames, height, width, 3))
        for i, frame in enumerate(video):
            tmp = cv2.resize(frame, (width, height))
            resized_video[i] = tmp
        return resized_video

class Pyramids():
    def __init__(self):
        self.levels = 4

    def build_gaussian_pyramid(self, frame):
        gauss = frame.copy()
        gauss_pyr = [gauss]

        for level in range(1, self.levels):
            gauss = cv2.pyrDown(gauss)
            gauss_pyr.append(gauss)
        return gauss_pyr

    def build_video_pyramid(self, frames):
        lap_video = []
        for i, frame in enumerate(frames):
            pyramid = self.build_gaussian_pyramid(frame)
            for j in range(self.levels):
                if i == 0:
                    lap_video.append(np.zeros((len(frames), pyramid[j].shape[0], pyramid[j].shape[1], 3)))
                lap_video[j][i] = pyramid[j]

        return lap_video
    
    def reconstruct_pyramid(self, step, video, num_frames,  width, height):
        resized_video = np.zeros((num_frames, height, width, 3))
        for i,frame in enumerate(video):
            for _ in range (0,step):
                frame = cv2.pyrUp(frame)
            resized_video[i] = frame

        return resized_video  


class Eulerian():

    def fft_filter(self, video, freq_min, freq_max, fps):
        fft=fftpack.fft(video,axis=0)
        frequencies = fftpack.fftfreq(video.shape[0], d=1.0 / fps)
        bound_low = (np.abs(frequencies - freq_min)).argmin()
        bound_high = (np.abs(frequencies - freq_max)).argmin()
        fft[:bound_low] = 0
        fft[bound_high:-bound_high] = 0
        fft[-bound_low:] = 0
        iff=fftpack.ifft(fft, axis=0)
        return np.abs(iff)
  
  
    def find_heart_rate(self, video, freq_min, freq_max, fps):
        fft = fftpack.rfft(video, axis=0)
        frequencies = fftpack.rfftfreq(fft.shape[0], d=1.0 / fps)
        fft_maximums = []
        # Bestimme die Amplitude an jedem Frequenzpunkt
        for i in range(fft.shape[0]):
            if freq_min <= frequencies[i] <= freq_max:
                fftMap = abs(fft[i])
                fft_maximums.append(fftMap.max())
            else:
                fft_maximums.append(0)

        peaks, properties = signal.find_peaks(fft_maximums)
        top_peak_freqs = []

        sorted_peaks = sorted(peaks, key=lambda x: fft_maximums[x], reverse=True)
        for peak in sorted_peaks[:100]:
            top_peak_freqs.append(frequencies[peak])

        if top_peak_freqs:
            average_freq = sum(top_peak_freqs) / len(top_peak_freqs)
            return average_freq * 60
        else:
            return None