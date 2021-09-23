import torch
#Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision import datasets
from torchvision.transforms import ToTensor
#train_data = datasets.MNIST(root = 'data',train = True,transform = ToTensor(),download = True)
#test_data = datasets.MNIST(root = 'data', train = False, transform = ToTensor())
train_data = datasets.FashionMNIST(root = 'data1',train = True,transform = ToTensor(),download = True)
test_data = datasets.FashionMNIST(root = 'data1', train = False, transform = ToTensor())
from torch.utils.data import DataLoader 
Batch_Size=100
Learning_Rate=2e-4
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=Batch_Size, 
                                          shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=Batch_Size, 
                                          shuffle=True, 
                                          num_workers=1),
}
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)    
    def forward(self, x):
    	x = self.conv1(x)
    	x = self.conv2(x)
    	x = x.view(x.size(0), -1)
    	output = self.out(x)
    	return output, x 
cnn = CNN()
cnn=cnn.to(device)
loss_func = nn.CrossEntropyLoss()  
from torch import optim
optimizer = optim.Adam(cnn.parameters(), lr = Learning_Rate)  
from torch.autograd import Variable
num_epochs = 10
def train(num_epochs, cnn, loaders):
    
    cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images).cuda()
            b_y = Variable(labels).cuda()
            output = cnn(b_x)[0]        
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()                # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))               
                pass
        
        pass
    
    
    pass
train(num_epochs, cnn, loaders)       
def test():
    # Test the model
    cnn.eval()    
    with torch.no_grad():
        correct = 0
        total = 0
        T_Acc=0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images.cuda())
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels.cuda()).sum().item() / float(labels.size(0))
            pass
        print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
    pass
Total_Accuracy=test()  
print("Total_Accuracy=",Total_Accuracy)  
