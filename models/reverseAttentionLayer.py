import math
import torch
from torch import nn
from models.ActivationFunction import GaussActivation, MaskUpdate
from models.weightInitial import weights_init

import pdb

# learnable reverse attention conv
class ReverseMaskConv(nn.Module):
    def __init__(self, inputChannels, outputChannels, kernelSize=4, stride=2, 
        padding=1, dilation=1, groups=1, convBias=False):
        super(ReverseMaskConv, self).__init__()

        self.reverseMaskConv = nn.Conv2d(inputChannels, outputChannels, kernelSize, stride, padding, \
            dilation, groups, bias=convBias)

        self.reverseMaskConv.apply(weights_init())

        # a, mu, sigma1, sigma2
        self.activationFuncG_A = GaussActivation(1.1, 1.0, 0.5, 0.5)
        self.updateMask = MaskUpdate(0.8)

        # pdb.set_trace()
        # (Pdb) a
        # self = ReverseMaskConv(
        #   (reverseMaskConv): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #   (activationFuncG_A): GaussActivation()
        #   (updateMask): MaskUpdate(
        #     (updateFunc): ReLU(inplace=True)
        #   )
        # )
        # inputChannels = 3
        # outputChannels = 64
        # kernelSize = 4
        # stride = 2
        # padding = 1
        # dilation = 1
        # groups = 1
        # convBias = False

    
    def forward(self, inputMasks):
        maskFeatures = self.reverseMaskConv(inputMasks)

        maskActiv = self.activationFuncG_A(maskFeatures)

        maskUpdate = self.updateMask(maskFeatures)

        # pdb.set_trace()
        # (Pdb) pp inputMasks.size()
        # torch.Size([1, 3, 1024, 1024])
        # (Pdb) maskActiv.size(), maskUpdate.size()
        # (torch.Size([1, 64, 512, 512]), torch.Size([1, 64, 512, 512]))

        return maskActiv, maskUpdate

# learnable reverse attention layer, including features activation and batchnorm
class ReverseAttention(nn.Module):
    def __init__(self, inputChannels, outputChannels, bn=False, activ='leaky', \
        kernelSize=4, stride=2, padding=1, outPadding=0,dilation=1, groups=1,convBias=False, bnChannels=512):
        super(ReverseAttention, self).__init__()

        # pdb.set_trace()
        # (Pdb) a
        # self = ReverseAttention()
        # inputChannels = 512
        # outputChannels = 512
        # bn = False
        # activ = 'leaky'
        # kernelSize = 4
        # stride = 2
        # padding = 1
        # outPadding = 0
        # dilation = 1
        # groups = 1
        # convBias = False
        # bnChannels = 1024


        self.conv = nn.ConvTranspose2d(inputChannels, outputChannels, kernel_size=kernelSize, \
            stride=stride, padding=padding, output_padding=outPadding, dilation=dilation, groups=groups,bias=convBias)
        
        self.conv.apply(weights_init())

        if bn:
            self.bn = nn.BatchNorm2d(bnChannels)
        
        if activ == 'leaky':
            self.activ = nn.LeakyReLU(0.2, False)
        elif activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'sigmoid':
            self.activ = nn.Sigmoid()
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'prelu':
            self.activ = nn.PReLU()
        else:
            pass
        # pdb.set_trace()


    def forward(self, ecFeaturesSkip, dcFeatures, maskFeaturesForAttention):
        # pdb.set_trace()

        nextDcFeatures = self.conv(dcFeatures)
        # (Pdb) nextDcFeatures.size()
        # torch.Size([1, 512, 16, 16])
        
        # note that encoder features are ahead, it's important tor make forward attention map ahead 
        # of reverse attention map when concatenate, we do it in the LBAM model forward function
        concatFeatures = torch.cat((ecFeaturesSkip, nextDcFeatures), 1)
        # (Pdb) concatFeatures.size()
        # torch.Size([1, 1024, 16, 16])
        
        outputFeatures = concatFeatures * maskFeaturesForAttention

        if hasattr(self, 'bn'):
            outputFeatures = self.bn(outputFeatures)
        if hasattr(self, 'activ'):
            outputFeatures = self.activ(outputFeatures)

        # pdb.set_trace()
        # (Pdb) ecFeaturesSkip.size(), dcFeatures.size(), maskFeaturesForAttention.size()
        # (torch.Size([1, 512, 16, 16]), torch.Size([1, 512, 8, 8]), torch.Size([1, 1024, 16, 16]))
        # (Pdb) outputFeatures.size()
        # torch.Size([1, 1024, 16, 16])

        return outputFeatures
