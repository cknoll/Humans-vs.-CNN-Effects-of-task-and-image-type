import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import ViTForImageClassification

#Image Size 224x224
class CustomVTrans(nn.Module): 
    def __init__(self): 
        super().__init__(); 
        self.model_name = "CustomVTrans"; 

    def forward(self, X): 
        pass; 

# ResNet Implementation

class BottleneckBlock(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, downsample=None, stride=1
    ):
        super().__init__()

        self.expansion = 4; 
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False); 
        self.bn1 = nn.BatchNorm2d(intermediate_channels);

        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False); 
        self.bn2 = nn.BatchNorm2d(intermediate_channels);

        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False); 
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion); 

        self.relu = nn.ReLU();
        self.downsample = downsample; 
        self.stride = stride;

    def forward(self, X):
        identity = X.clone(); 
        
        X = self.relu(self.bn1(self.conv1(X))); 
        X = self.relu(self.bn2(self.conv2(X))); 
        X = self.bn3(self.conv3(X)); 

        if self.downsample is not None:
            identity = self.downsample(identity);

        X += identity;
        X = self.relu(X); 
        return X;  


class BasicBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, stride=1, downsample=None): 
        super().__init__(); 
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False);
        self.bn1 = nn.BatchNorm2d(out_channels); 
        self.relu = nn.ReLU(); 
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False); 
        self.bn2 = nn.BatchNorm2d(out_channels); 

        self.downsample=downsample; 
        self.stride = stride; 
    
    def forward(self, X): 
        identity = X.clone();  
    
        X = self.relu(self.bn1(self.conv1(X))); 
        X = self.bn2(self.conv2(X)); 
        
        if self.downsample is not None: 
            identity = self.downsample(identity); 
        
        X += identity; 
        X = self.relu(X); 
        return X; 

class ResNet(nn.Module): 
    def __init__(self, block, layers, model_name): 
        super().__init__(); 
        self.model_name = model_name;

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=2, padding=3); 
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1); 
        self.relu = nn.ReLU(); 
        
        if block == BasicBlock: 
            self.layer1 = self._make_basic_layer(block, 64, 64, layers[0], stride=1); 
            self.layer2 = self._make_basic_layer(block, 64, 128, layers[1], stride=2); 
            self.layer3 = self._make_basic_layer(block, 128, 256, layers[2], stride=2); 
            self.layer4 = self._make_basic_layer(block, 256, 512, layers[3], stride=2); 
            linear_input = 512; 
        elif block == BottleneckBlock: 
            self.layer1 = self._make_bottleneck_layer(block, 64, 64, layers[0], stride=1); 
            self.layer2 = self._make_bottleneck_layer(block, 64*4, 128, layers[1], stride=2); 
            self.layer3 = self._make_bottleneck_layer(block, 128*4, 256, layers[2], stride=2); 
            self.layer4 = self._make_bottleneck_layer(block, 256*4, 512, layers[3], stride=2); 
            linear_input = 2048; 
        else: 
            print("Unknown Block type for ResNet"); 
            quit(); 

       
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)); 
        self.fc = nn.Linear(linear_input, 365); 
    def forward(self, X): 
        # Annotations made for ResNet with Basic Blocks, assuming an initial input size of: bs x 3 x 224 x 224
        #Initial Downsampling
        X = self.conv1(X);          # bs x 64 x 112 x 112
        X = self.maxpool(X);        # bs x 64 x 56 x 56 
        X = self.relu(X);  
        
        # ResNet Layers
        X = self.layer1(X);         # bs x 64 x 56 x 56
        X = self.layer2(X);         # bs x 128 x 28 x 28
        X = self.layer3(X);         # bs x 256 x 14 x 14
        X = self.layer4(X);         # bs x 512 x 7 x 7

        X = self.avgpool(X);            # bs x 512 x 1 x 1
        X = X.reshape(X.shape[0], -1);  # bs x 512
        X = self.fc(X);                 # bs x 365 
        return X; 

    def _make_basic_layer(self, block, in_channels, out_channels, blocks, stride=1): 
        downsample=None; 
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), 
                nn.BatchNorm2d(out_channels)
            );
        layers = [];
        layers.append(block(in_channels, out_channels, stride, downsample)); 
        in_channels = out_channels; 

        for _ in range(blocks - 1): 
            layers.append(block(in_channels, out_channels)); 

        return nn.Sequential(*layers); 

    def _make_bottleneck_layer(self, block, in_channels, intermediate_channels, blocks, stride=1):
        downsample = None;
        layers = [];
        
        if stride != 1 or in_channels != intermediate_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, intermediate_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(intermediate_channels * 4),
            ); 
        layers.append(block(in_channels, intermediate_channels, downsample, stride));
        in_channels = intermediate_channels * 4
        for i in range(blocks - 1):
            layers.append(block(in_channels, intermediate_channels)); 
        return nn.Sequential(*layers); 


class CustomCNN(nn.Module): 
    def __init__(self): 
        super().__init__();
        self.model_name = "CustomCNN"; 

        self.conv1 = nn.Conv2d(3, 32, 3, padding=(1, 1)); 
        self.conv2 = nn.Conv2d(32, 128, 3, padding=(1, 1)); 
        self.conv3 = nn.Conv2d(128, 512, 3, padding=(1, 1)); 
#        self.conv4 = nn.Conv2d(512, 2048, 3, padding=(1, 1));   
        self.fc1 = nn.Linear(512 * 28 * 28, 365); 

    def forward(self, X):
        X = F.max_pool2d(F.relu(self.conv1(X)), (2, 2)); # -> X.shape: bs x 32 x 112 x 112
        X = F.max_pool2d(F.relu(self.conv2(X)), (2, 2)); # -> X.shape: bs x 128 x 56 x 56
        X = F.max_pool2d(F.relu(self.conv3(X)), (2, 2)); # -> X.shape: bs x 512 x 28 x 28
#        X = F.max_pool2d(F.relu(self.conv4(X)), (2, 2)); # -> X.shape: bs x 2048 x 14 x 14
        X = X.reshape(-1, 512 * 28 * 28); 
        X = self.fc1(X); 
        return X; 

class CustomVGG16ZeroPad(nn.Module): 
    def __init__(self): 
        super().__init__(); 
        self.model_name = "CustomVGG16ZeroPad"; 

        self.conv1_1 = nn.Conv2d(3, 64, 3); 
        self.conv1_2 = nn.Conv2d(64, 64, 3); 
        
        self.conv2_1 = nn.Conv2d(64, 128, 3); 
        self.conv2_2 = nn.Conv2d(128, 128, 3); 
    
    def forward(self, X): 
        #(N, 3, 224, 224)
        print(X.shape); 
        X = F.relu(self.conv1_1(X));    
        #(N, 64, 222, 222)
        print(X.shape); 
        X = F.max_pool2d(F.relu(self.conv1_2(X))); 
        #(N, 64, 110, 110)
        print(X.shape); 
        X = F.relu(self.conv2_1(X)); 
        #(N, 128, 108, 108)
        print(X.shape); 
        X = F.max_pool2d(F.relu(self.conv2_2(X))); 
        #(N, 128, 53, 53)
        print(X.shape); 
          
        #(N, 128, 51, 51)
        #(N, 128, 49, 49)
        #(N, 128, 47, 47)

class CustomVGG16(nn.Module): 
    def __init__(self): 
        super().__init__(); 
        self.model_name = "CustomVGG16"; 

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1); 
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1); 
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1); 
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1); 

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1); 
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1); 
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1); 

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1); 
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1); 
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1); 

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1); 
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1); 
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1); 

        self.fc1 = nn.Linear(7 * 7 * 512, 365); 
    def forward(self, X):
        X = F.relu(self.conv1_1(X)); 
        X = F.max_pool2d(F.relu(self.conv1_2(X)), (2, 2)); 
        
        X = F.relu(self.conv2_1(X)); 
        X = F.max_pool2d(F.relu(self.conv2_2(X)), (2, 2)); 

        X = F.relu(self.conv3_1(X)); 
        X = F.relu(self.conv3_2(X)); 
        X = F.max_pool2d(F.relu(self.conv3_3(X)), (2, 2)); 

        X = F.relu(self.conv4_1(X)); 
        X = F.relu(self.conv4_2(X)); 
        X = F.max_pool2d(F.relu(self.conv4_3(X)), (2, 2)); 

        X = F.relu(self.conv5_1(X)); 
        X = F.relu(self.conv5_2(X)); 
        X = F.max_pool2d(F.relu(self.conv5_3(X)), (2, 2)); 

        X = X.reshape(X.shape[0], -1); 
        X = self.fc1(X); 
        return X; 

class CustomVGG16ZeroPad(nn.Module): 
    def __init__(self): 
        super().__init__();  
        self.model_name = "CustomVGG16ZeroPad";

        self.conv1_1 = nn.Conv2d(3, 64, 3);
        self.conv1_2 = nn.Conv2d(64, 64, 3);
        
        self.conv2_1 = nn.Conv2d(64, 128, 3);
        self.conv2_2 = nn.Conv2d(128, 128, 3);
    
        self.conv3_1 = nn.Conv2d(128, 256, 3); 
        self.conv3_2 = nn.Conv2d(256, 256, 3); 
        self.conv3_3 = nn.Conv2d(256, 256, 3); 

        self.conv4_1 = nn.Conv2d(256, 512, 3); 
        self.conv4_2 = nn.Conv2d(512, 512, 3); 
        self.conv4_3 = nn.Conv2d(512, 512, 3); 
       
        self.conv5_1 = nn.Conv2d(512, 512, 3); 
        self.conv5_2 = nn.Conv2d(512, 512, 3); 
        self.conv5_3 = nn.Conv2d(512, 512, 3); 

        self.final_conv = nn.Conv2d(512, 2048, 2); 

        self.out = nn.Linear(2048, 365); 

    def forward(self, X): 
        X = F.relu(self.conv1_1(X));
        X = F.max_pool2d(F.relu(self.conv1_2(X)), 2);

        X = F.relu(self.conv2_1(X));
        X = F.max_pool2d(F.relu(self.conv2_2(X)), 2);
        
        X = F.relu(self.conv3_1(X)); 
        X = F.relu(self.conv3_2(X)); 
        X = F.max_pool2d(F.relu(self.conv3_3(X)), 2); 
        
        X = F.relu(self.conv4_1(X)); 
        X = F.relu(self.conv4_2(X)); 
        X = F.max_pool2d(F.relu(self.conv4_3(X)), 2); 

        X = F.relu(self.conv5_1(X)); 
        X = F.relu(self.conv5_2(X)); 
        X = F.relu(self.conv5_3(X)); 

        X = F.relu(self.final_conv(X)); 

        X = X.reshape(X.shape[0], -1); 
        X = self.out(X);
        return X; 


class ViTB16(nn.Module):
    def __init__(self): 
        super().__init__(); 
        self.model_name = "ViTB16";  
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=365); 
        self.vit.heads = nn.Sequential(
            nn.Linear(768, 365)
        ); 

    def forward(self, X): 
        outputs = self.vit(pixel_values=X); 
        return outputs.logits; 

def construct_model(m_name): 
    if m_name == "CustomCNN": 
        return CustomCNN(); 
    elif m_name == "CustomVGG16":
        return CustomVGG16(); 
    elif m_name == "ResNet34": 
        return ResNet(BasicBlock, [3, 4, 6, 3], "ResNet34"); 
    elif m_name == "ResNet50": 
        return ResNet(BottleneckBlock, [3, 4, 6, 3], "ResNet50"); 
    elif m_name == "ResNet101": 
        return ResNet(BottleneckBlock, [3, 4, 23, 3], "ResNet101");
    elif m_name == "ResNet152": 
        return ResNet(BottleneckBlock, [3, 8, 36, 3], "ResNet152");
    elif m_name == "ViTB16": 
        return ViTB16();  
    elif m_name == "CustomVGG16ZeroPad": 
        return CustomVGG16ZeroPad(); 
    else: 
        return None;









