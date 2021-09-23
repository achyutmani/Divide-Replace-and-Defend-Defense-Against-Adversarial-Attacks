import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder, MNIST
from torchvision import transforms
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import random 
SEED = 1234 # Initialize seed 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
writer = SummaryWriter()
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))])
batch_size = 4096*2
data_loader = torch.utils.data.DataLoader(MNIST('MNIST_Dataset', train=True, download=True, transform=transform),
                                          batch_size=batch_size, shuffle=True)
class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()
class generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)
generator = generator().cuda()
discriminator = discriminator().cuda()
#print(generator)
#print(discriminator)
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)
def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).cuda()
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).cuda())
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()
def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    d_optimizer.zero_grad()

    # train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).cuda())
    
    # train with fake images
    z = Variable(torch.randn(batch_size, 100)).cuda()
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).cuda())
    
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()
num_epochs = 10
n_critic = 5
display_step = 50
MODEL_SAVE_PATH='generator_state.pth'
generator.load_state_dict(torch.load(MODEL_SAVE_PATH))
#torch.save(generator.state_dict(), 'generator_state.pth')
#z = Variable(torch.randn(1, 100)).cuda()
#labels=torch.LongTensor(0).cuda()
#labels = torch.LongTensor([i for i in range(10) for _ in range(10)]).cuda()
#images = generator(z, labels).unsqueeze(1)
#grid = make_grid(images, nrow=10, normalize=True)
#grid=grid.detach().cpu()
#fig, ax = plt.subplots(figsize=(10,10))
#ax.imshow(grid.permute(1, 2, 0).data, cmap='binary')
#ax.axis('off')
#plt.imshow(images)
#plt.show()
from flopth import flopth
def matrix_to_image(spec, eps=1e-9):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
  return spec_scaled
def generate_digit(generator, digit):
    z = Variable(torch.randn(1, 100)).cuda()
    label = torch.LongTensor([digit]).cuda()
    img = generator(z, label).data.cpu()
    img = 0.5 * img + 0.5
    return img 
M=generate_digit(generator, 9)  
M=matrix_to_image(np.array(M), eps=1e-9) 
#M=np.array(M)
#M = ((M - M.min()) * (1/(M.max() - M.min()) * 255))
print(M.shape)
M=torch.from_numpy(M)
M=transforms.ToPILImage()(M)
plt.imshow(M,cmap='gray', vmin=0, vmax=255)
#plt.savefig('GAN_9.jpg')
#plt.show() 
#def FLOPS_Count(Model):
#    #warnings.filterwarnings("ignore")
#    sum_flops1 = flopth(Model, in_size=[[1,100]],[1])
#    print("Number of FLOPS M1=",sum_flops1)
#print("Number of FLOPs for PAM Module=\n")
#FLOPS_Count(generator)

