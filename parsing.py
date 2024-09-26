import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace = True)
        )
        

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.Sigmoid(self.conv(x))

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        ################################
        filter = 32

        filter_two = filter*2
        filter_three = filter_two*2
        filter_four = filter_three*2
        filter_five = filter_four*2

        self.inc = (DoubleConv(n_channels, filter))
        self.down1 = (Down(filter, filter_two))
        self.down2 = (Down(filter_two, filter_three))
        self.down3 = (Down(filter_three, filter_four))

        factor = 2 if bilinear else 1

        self.down4 = (Down(filter_four, filter_five // factor))
        self.up1 = (Up(filter_five, filter_four // factor, bilinear))
        self.up2 = (Up(filter_four, filter_three // factor, bilinear))
        self.up3 = (Up(filter_three, filter_two // factor, bilinear))
        self.up4 = (Up(filter_two, (filter), bilinear))
        self.outc = (OutConv(filter, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits




class FaceParsing():
    def __init__(self):
        self.model_path = '/Users/philipprenner/Desktop/Work/FaceParsing/face_segmentation_model_weights.pt'
        self.model = UNet(n_channels=1, n_classes=1) 
        print(self.model)
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        

    def parse_frame(self, video):
        device = torch.device("mps")
        self.model.to(device)
        self.model.eval()
        
        masks = []
        face_segmented_video = []
        for index, frame in enumerate(video):
            tensor = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            tensor = cv2.resize(tensor, (128, 128))

            tensor = tensor/np.amax(tensor)
            tensor = torch.tensor(tensor)
            tensor = torch.unsqueeze(torch.unsqueeze(tensor,0),0)
            tensor = tensor.to(torch.float32)
            tensor = tensor.to(device)

            prediction = self.model(tensor)
            prediction = prediction.squeeze()
            prediction = prediction.squeeze()
            prediction = prediction.cpu().detach().numpy()

            threshold = 0.5
            prediction[prediction < threshold] = 0
            prediction[prediction > threshold] = 1
            
            kernel = np.ones((3, 5), np.uint8) 
            prediction = cv2.dilate(prediction, kernel, iterations=1)
            
            mask_calculation = np.zeros((*prediction.shape,3))
            mask_calculation[:,:,0] = prediction
            mask_calculation[:,:,1] = prediction
            mask_calculation[:,:,2] = prediction

            mask_calculation = cv2.resize(mask_calculation, (frame.shape[1], frame.shape[0]))

            face_segment = mask_calculation*frame
            face_segment = face_segment.astype(np.float32)
            face_segment = cv2.cvtColor(face_segment, cv2.COLOR_BGR2RGB)

            face_segmented_video.append(face_segment)
            
            masks.append(mask_calculation)

            

        masks = np.array(masks)     
        video = np.array(video)     

        if masks is None:
            pass                
        
        return masks, face_segmented_video



    def plotter(self, mask, img):
            mask_plot = np.zeros((720, 1280, 4), dtype=np.uint8)  
            mask_plot[..., 0] = mask * 255  
            mask_plot[..., 1] = 0           
            mask_plot[..., 2] = 0          
            mask_plot[..., 3] = mask * 128  

            mask_calculation = np.zeros((*mask.shape,3))
            mask_calculation[:,:,0] = mask
            mask_calculation[:,:,1] = mask
            mask_calculation[:,:,2] = mask
            
            if img.ndim == 2: 
                img_rgb = np.stack((img,)*3, axis=-1) 
            else: masked_image = img.astype(int) * mask_calculation.astype(int)

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img, cmap='gray')
            axs[0].set_title('Originalbild')
            axs[0].axis('off')
            axs[1].imshow(masked_image)  
            axs[1].set_title('Überlagerung')
            axs[1].axis('off')
            axs[2].imshow(mask_plot, cmap='gray')
            axs[2].set_title('Maske')
            axs[2].axis('off')
            plt.show()


