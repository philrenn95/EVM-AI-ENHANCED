
import cv2
import numpy
import tqdm
from datetime import datetime, timedelta
import numpy as np
import torch
import os
import FITBIT_GROUNDTRUTH 

class TensorFromVideos():

    def data_import(self):
            """
            Imports the file names 
            """
            self.RgbPath = os.path.join("/Volumes/Untitled/EVM", "RGB", self.ChosenParticipant + "-RGB/")
            participants = os.listdir(os.path.join(os.getcwd()))

            self.RgbData = os.listdir(self.RgbPath)
            self.RgbData = [file for file in self.RgbData if file.endswith(".avi")]

    def fit_bit(self):
            """
            Retrieves and processes Fitbit ground truth data for the chosen participant.

            Returns:
            numpy.ndarray: Array containing the ground truth heart rate values.
            """
            
            fit_bit_data = FITBIT_GROUNDTRUTH.FITBIT_GROUNDTRUTH()
            Groundtruth = fit_bit_data[self.ChosenParticipant]
            self.GroundtruthTimesstamps = np.zeros(len(Groundtruth[:,0]))

            for index, iterator in enumerate(Groundtruth[:,0]):
                self.GroundtruthTimesstamps[index] = self.time_to_seconds(iterator)

            GroundtruthHeartrate = Groundtruth[:,1].astype(int)
        
            return(GroundtruthHeartrate)

    def time_to_seconds(self, time_str):
            """
            Converts a time string in the format "hours:minutes:seconds" into the 
            corresponding number of seconds since midnight.

            Parameters:
            time_str (str): The string representing time in the format "hours:minutes:seconds".

            Returns:
            float: The number of seconds since midnight corresponding to the given time.
            """
            time_obj = datetime.strptime(time_str, "%H:%M:%S")
            HourInSeconds = time_obj.hour * 3600
            MinuiteInSeconds = time_obj.minute * 60
            Seconds = time_obj.second
            tmp = HourInSeconds + MinuiteInSeconds + Seconds
            return  tmp


    def save_training_data(self):
            ParticipantList = ["UBJUW", "AUCTQ", "AXORW", "EIRZB", "ELXXU", "HZEZJ",
                                    "NCDXU", "OMEGC", "PIUYU", "PZQGH", "RIIQO", "RKPNY",
                                    "RXODG", "SIUJQ", "SJFSC","UROJF", "USXZQ", "WBBCD",
                                    "YREUM", "YXYCW"]
            
            for Participant in ParticipantList:
                print(f"PARTICIPANT   :   {Participant}")
                self.ChosenParticipant = Participant
                self.data_import()
                ground_truth_pulse_file = self.fit_bit()

                for file_name in tqdm(self.RgbData):
                    if file_name[0] != "R":
                        continue
                    cap = cv2.VideoCapture(self.RgbPath + file_name)
                    TimestampLowerBound = file_name[21:29].replace("-",":").replace('_', ':')
                    time_obj = datetime.strptime(TimestampLowerBound, "%H:%M:%S")

                    if not cap.isOpened():
                        print("Fehler beim Öffnen der Videodatei")
                        continue

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                    video = []
                    frame_window = int(fps * self.window_length)
                    length = 0
                    counter = 0

                    #iterate over the provided video file

                    if num_frames == 0:
                        continue
                        
                    while True:
                        ret, frame = cap.read()

                        if not ret: break
                        else: length = length + 1
                        video.append(frame)
                            
                        if length % frame_window == 0:
                            time_obj = time_obj + timedelta(seconds=self.window_length)
                            TimestampUpperBound = time_obj.strftime("%H:%M:%S")

                            IndexLowerBound = int(np.argmin(np.abs(self.GroundtruthTimesstamps - self.time_to_seconds(TimestampLowerBound))))
                            IndexUpperBound = int(np.argmin(np.abs(self.GroundtruthTimesstamps - self.time_to_seconds(TimestampUpperBound))))
                            if IndexLowerBound == IndexUpperBound:
                                ground_truth_pulse = ground_truth_pulse_file[IndexLowerBound]
                            else:
                                ground_truth_pulse = np.mean(ground_truth_pulse_file[IndexLowerBound:IndexUpperBound])

                            counter = counter + 1

                            for i, frame in enumerate(video):
                                video[i] = video[i]/np.amax(video[i])

                            video = np.array(video).astype(np.float32)
                            masks, video  = self.face_parsing.parse_frame(video=video)
                            video_training = []
                            false_counter = 0

                            for i in range(len(video)):
                                frame = np.array(video[i])
                                mask = np.array(masks[i,:,:,0])
                                ys, xs = np.nonzero(mask)
                                if np.amax(mask) == 0: 
                                    face = frame[y_min:y_max+1, x_min:x_max+1]
                                    false_counter = false_counter + 1
                                else:                 
                                    x_min, x_max = np.min(xs), np.max(xs)
                                    y_min, y_max = np.min(ys), np.max(ys)
                                    face = frame[y_min:y_max+1, x_min:x_max+1]
                                
                                face_resized = cv2.resize(face[:,:,:2], (64, 64))
                                video_training.append(face_resized)
                            
                            video_training = np.array(video_training)
                            video_training[0,0,0,0] = fps
                            video_training[1,0,0,0] = ground_truth_pulse

                            video_training = torch.tensor(video_training)

                            if np.isnan(ground_truth_pulse):
                                print(f"NAN Error at |{file_name} with groundtruth {ground_truth_pulse}")

                            elif int(ground_truth_pulse) == 0:
                                print(f"0 Error at |{file_name} with groundtruth {ground_truth_pulse}")

                            elif false_counter <= (frame_window//2):
                                save_path = f"data/Train64/{file_name}_tensor_{counter}.pt"
                                torch.save(video_training, save_path)

                            else:
                                print(f"Error at |{file_name} with groundtruth {ground_truth_pulse}")
                                pass

                            TimestampLowerBound = TimestampUpperBound
                            video = []


if __name__ == "__main__":
    generator = TensorFromVideos()
    generator.save_training_data()