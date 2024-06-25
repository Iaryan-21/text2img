import torch 
import torch.nn as nn
import torch.nn.functional as F 

class StageIGenerator(nn.Module):
    def __init__(self, text_embedding_dim, noise_dim, image_size, channels):
        super(StageIGenerator,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_embedding_dim+ noise_dim, 128 * (image_size//4)*(image_size//4)),
            nn.ReLU(True)
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        def forward(self, text_embedding, noise):
            x = torch.cat((text_embedding, noise), dim=1)
            x = self.fc(X)
            x = x.view(x.size(0), 128, image_size//4, image_size//4)
            x = x.upsample(x)
            return x

class StageIDiscriminator(nn.Module):
    def __init__(self, image_size, channels, text_embedding_dim):
        super(StageIDiscriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(128*(image_size//4)*(image_size//4) + text_embedding_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, text_embedding):
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, text_embedding), dim=1)
        x = self.fc(X)
        return x

# noise_dim = 100
# image_size = 64
# channels = 3
# hidden_dim = 256 
# embedding_dim = 128

# stage1_gan = StageIGenerator(hidden_dim, noise_dim, image_size, channels)
# stage1_disc = StageIDiscriminator(image_size, channels, hidden_dim)
# print(stage1_gan)
# print(stage1_disc)

class StageIIGenerator(nn.Module):
    def __init__(self, text_embedding_dim, noise_dim, image_size, channels):
        super(StageIIGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_embedding_dim+noise_dim, 128*(image_size//4)*(image_size//4)),
            nn.ReLU(True)
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, text_embedding, noise):
        x = torch.cat((text_embedding, noise), dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 128, image_size//4, image_size//4)
        x = self.upsample(x)
        return x

class StageIIDiscriminator(nn.Module):
    def __init__(self, image_size, channels, text_embedding_dim):
        super(StageIIDiscriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3, inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(128*(image_size//4)*(image_size//4) + text_embedding_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, text_embedding):
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, text_embedding), dim=1)
        x = self.fc(x)
        return x

# stage2_gen = StageIIGenerator(hidden_dim, noise_dim, image_size*4, channels)
# stage2_disc = StageIIDiscriminator(image_size*4, channels, hidden_dim)
# print(stage2_gen)
# print(stage2_disc)