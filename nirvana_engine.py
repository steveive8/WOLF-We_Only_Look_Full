import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

# 2021 July
# Steve Ive
# MIT Licence
# WOLF: We Only Look Full
# - With Nirvana Engine and Detection



#model input 단만 weight 다르게 해서
#나중에는 다 합하는 생각도 해봤는데
#어차피 이미지 사이즈 다양해서 의미없을듯




#Concept
#Nirvana doesn't use Dropout.
#Nirvana believes the power of Banila Neural Network. (Fully-Connected)
#Nirvana shows the Best capability made of the Simple Convolutional and Linear Layers.

#The Power of Nirvana Detection is on the Full-to-Part Sizing for unique distribution detecting.
#Nirvana Detection call it the WOLF, We Only Look Full.
#WOLF is inspired of the person looking the object, 
#which is focusing unique object on the as possible as plain background compare the object, after recognizing the full background.

#Specifically, it was inspired from the bathroom of Steve Ive, when I shower and think about how human see the object.
#It was the Shampoo of my bathroom. That was the object of the inspiration.

class NirvanaNet(nn.Module):
    def __init__(self, inSize = 3, outSize = 10, trainmode = True):
        super().__init__()
        self.inSize = inSize
        self.outSize = outSize
        self.trainmode = trainmode
        self.expandGamma = nn.Linear(1, 32)#(1, image_size)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size= 2, stride = 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride = 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride = 2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride = 2)
        )
        self.layer6 = nn.Sequential(
            nn.Linear(64 * 2 * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Linear(128, 10)
        )
        self.layers = (
            self.layer1, 
            self.layer2, 
            self.layer3, 
            self.layer4, 
            self.layer5, 
            self.layer6, 
            self.layer7, 
            self.layer8
        )
        self.ConvLayers = (
            self.layer1, 
            self.layer2, 
            self.layer3, 
            self.layer4, 
            self.layer5, 
        )
        self.FCLayers = (
            self.layer6, 
            self.layer7, 
            self.layer8
        )
        self.weight_init()

    def layer0(self, x):
        out = ((x - x.mean()) / x.std())
        out = self.expandGamma(x)
        return out

    def forward(self, x):
        if self.trainmode:
            print(x.shape)
            out = self.layer0(x)
            print(out.shape)
            out = self.layer1(out)
            print(out.shape)
            out = self.layer2(out)
            print(out.shape)
        else:
            out = x
        out = self.layer3(out)
        print(out.shape)
        out = self.layer4(out)
        print(out.shape)
        out = self.layer5(out)
        print(out.shape)
        out = self.layer6(out)
        print(out.shape)
        out = self.layer7(out)
        out = self.layer8(out)
        return out

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.expandGamma.weight, 1.3)
        nn.init.constant_(self.expandGamma.bias, 0)






                


class NirvanaDetection(NirvanaNet):
    #__init__()
        #get classes -> function2
        #create box of the size of 2 / 3 of full image, 2 / 3 size of the first box, 2 / 3 size of the second box
    #inference()
        #and set height and widthimage standarization and multiply weights and add bias of expandGamma, and start inference
    #mle_looper()
        #loop for image (scan diagonally first for lefttop to rightbottom, righttop to leftbottom => and move the detector with keeping the shape of diagonal)
        #uniqueDistribution()
            #calculate unique distribution => normalize
            #=> unique distribution => normalize
            #=> unique distribution => normalize
        #MLE()
            #with listed mles, detect with nirvna engine
            #if not >= threshold 
                # => list initialize
            #else 
                #-> detect()
                    # return label and get the pos of bounding box and class name

    #about unique distribution
    #unique distribution can be consist of two things
        #- the shape
        #- the color
    #nirvana engine detect the unique shape and color from background as possible(object that is more than unique than background).

    def __init__(self, width = 640, height = 480, distributionThreshold = 0.55, detectThreshold = 0.85):
        super(NirvanaDetection, self).__init__()
        self.height = height
        self.width = width
        self.threshold = distributionThreshold
        self.detectThrehsold = detectThreshold
        self.box1size = [np.round(self.width * 2 / 3), np.round(self.height * 2 /3)]
        self.box2size = [np.round(self.box1size[0] * 2 / 3), np.round(self.box1size[1] * 2 / 3)]
        self.box3size = [np.round(self.box2size[0] * 2 / 3), np.round(self.box2size[1] * 2 / 3)]
        self.box4size = [np.round(self.box3size[0] * 2 / 3), np.round(self.box3size[1] * 2 / 3)]
        self.box5size = [np.round(self.box4size[0] * 2 / 3), np.round(self.box4size[1] * 2 / 3)]
        self.boxsizes = [self.box1size, self.box2size, self.box3size, self.box4size, self.box5size]
        self.nirvanaStride = [np.round(self.width / 10), np.round(self.height / 10)]
        self.mles = [[], []]
        self.trainmode = False
        #mles[0] => left,
        #mles[1] => right

    def inference(self, image):
        image.height = self.height
        image.width = self.width
        out = ((image - image.mean()) / image.std())
        image = self.layer0(out)
        return self.mle_looper(image)
        

    def mle_looper(self, image):
        for sizeindex, boxsize in enumerate(self.boxsizes):
            for left in range(10):
                for i in range(10):
                    stride = i * self.nirvanaStride
                    image = image[stride[1] : boxsize[1] + stride[1], left * stride[0] + stride[0] : boxsize[0] + stride[0]]
                    self.uniqueDistribution(image, 0, sizeindex, left, i)
                self.MLE(0, image)

                right = abs(left - 10)

                for i in range(10):
                    stride = i * self.nirvanaStride
                    image = image[stride[1] : stride[1] + boxsize[1], right * stride[0] - stride[0]: stride[0] + boxsize[0]]
                    self.uniqueDistribution(image, 1, sizeindex, right, i)
                self.MLE(1, image)
            

    def uniqueDistribution(self, x, leftOrRight, widthStride, heightStride):
        value = self.layer1(x)
        value = self.layer2(value)
        result = {'value': value, 'widthStride': widthStride, 'heightStride': heightStride}
        return self.mles[leftOrRight].append(result)

    def MLE(self, leftOrRight, image):
        for imgs in self.mles[leftOrRight]:
            if imgs['value'] >= self.threshold:
                stride = imgs['heightStride'] * self.nirvanaStride
                image = image[stride[1] : self.boxsizes[imgs['sizeindex']] + stride[1], imgs['widthStride'] * stride[0] + stride[0] : self.boxsizes[imgs['sizeindex']] + stride[0]]
                return self.detect(image, imgs)
            else:
                self.mles[leftOrRight] = []

    def generateBoundingBox(self, prop):
        pass
        
    def detect(self, x, prop):
        results = self.forward(x)
        if [results > self.detectThrehsold].float().sum() >= 1:
            self.generateBoundingBox(prop)
            return torch.argmax(results, 1), results




class Utils(nn.Module):
    def __init__(self, model, learning_rate = 0.001 , epochs = 10, dataset = 'cifar10', train_batch_size = 512):
        super().__init__()
        self.deviceinit()
        self.device
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.losses = []
        self.trainloader = None
        self.testloader = None
        self.accs = []
        self.classes = None
        #self.lr_sche = optim.lr_scheduler.StepLR(self.optim, step_size=5, gamma=0.9)
        if dataset == 'cifar10':
            self.cifar10()

    def deviceinit(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
    
    def optim(self):
        return optim.Adam(self.model.parameters(), lr = self.learning_rate)

    def cifar10(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.VOCDetection(root='./voc', image_set='train',
                                                download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch_size, shuffle=True, num_workers=0)

        testset = torchvision.datasets.VOCDetection(root='./voc', image_set='val', download=True, transform=transform)

        self.testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

        self.classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tv/monitor']

    def train(self):
        
        print('Learning Started...')
        print()
        print()
        print('WOLF: We Only Look Full, 1.0')
        print('Powered By Nirvana Net')
        print('Copyright By Steve Ive, 2021 July')
        print()

        self.model.train()

        for epoch in range(self.epochs):

            print('{} Epoch Started'.format(epoch + 1))

            epoch_losses = 0.0
            minibatch_loss = 0.0
            #self.lr_sche.step()

            for i, data in enumerate(self.trainloader):

                X , Y = data
                X = X.to(self.device)
                Y = Y.to(self.device)

                #prediction
                pred = self.model(X)

                #cost
                cost = F.cross_entropy(pred, Y).to(self.device)

                #Reduce the cost
                self.optim.zero_grad()
                cost.backward()
                self.optim.step()

                minibatch_loss += cost
                epoch_losses += cost

                if i % 30 == 29:
                    minibatch_loss = minibatch_loss / 30
                    print('   Epoch {} / {} - MiniBatch: {} / {}, MiniBatch(30)_Loss: {:.6f}'.format(epoch + 1, self.epochs, i, len(self.trainloader), minibatch_loss))
                    minibatch_loss = 0.0

            epoch_losses = epoch_losses / len(self.trainloader)
            self.losses.append(epoch_losses)
            print('')
            print('{} Epoch Finished. Epoch_Loss: {}'.format(epoch + 1, epoch_losses))
            
        
        print('Learning Finished. Total Loss: {}'.format(self.losses.mean()))


model = NirvanaNet(trainmode=True)
utils = Utils(model)

model.train()
utils.train()

detector = NirvanaDetection()