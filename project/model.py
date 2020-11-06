"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 02日 星期一 17:49:55 CST
# ***
# ************************************************************************************/
#

import os
import sys
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from tqdm import tqdm
from torchvision import models

import pdb

# The following comes from https://github.com/Vious/LBAM_Pytorch
# Image Inpainting With Learnable Bidirectional Attention Maps
# Thanks authors a lot.

class ImagePatchModel(nn.Module):
    """ImagePatch Model."""

    def __init__(self, inputChannels = 4, outputChannels = 3):
        """Init model."""
        super(ImagePatchModel, self).__init__()

        # default kernel is of size 4X4, stride 2, padding 1, 
        # and the use of biases are set false in default ReverseAttention class.
        self.ec1 = ForwardAttention(inputChannels, 64, bn=False)
        self.ec2 = ForwardAttention(64, 128)
        self.ec3 = ForwardAttention(128, 256)
        self.ec4 = ForwardAttention(256, 512)

        for i in range(5, 8):
            name = 'ec{:d}'.format(i)
            setattr(self, name, ForwardAttention(512, 512))
        
        # reverse mask conv
        self.reverseConv1 = ReverseMaskConv(3, 64)
        self.reverseConv2 = ReverseMaskConv(64, 128)
        self.reverseConv3 = ReverseMaskConv(128, 256)
        self.reverseConv4 = ReverseMaskConv(256, 512)
        self.reverseConv5 = ReverseMaskConv(512, 512)
        self.reverseConv6 = ReverseMaskConv(512, 512)

        self.dc1 = ReverseAttention(512, 512, bnChannels=1024)
        self.dc2 = ReverseAttention(512 * 2, 512, bnChannels=1024)
        self.dc3 = ReverseAttention(512 * 2, 512, bnChannels=1024)
        self.dc4 = ReverseAttention(512 * 2, 256, bnChannels=512)
        self.dc5 = ReverseAttention(256 * 2, 128, bnChannels=256)
        self.dc6 = ReverseAttention(128 * 2, 64, bnChannels=128)
        self.dc7 = nn.ConvTranspose2d(64 * 2, outputChannels, kernel_size=4, stride=2, padding=1, bias=False)

        self.tanh = nn.Tanh()

    def forward(self, inputImgs, masks):
        """Forward."""

        # pdb.set_trace()
        # (Pdb) pp inputImgs.size(), masks.size()
        # (torch.Size([1, 4, 1024, 1024]), torch.Size([1, 3, 1024, 1024]))

        ef1, mu1, skipConnect1, forwardMap1 = self.ec1(inputImgs, masks)
        ef2, mu2, skipConnect2, forwardMap2 = self.ec2(ef1, mu1)
        ef3, mu3, skipConnect3, forwardMap3 = self.ec3(ef2, mu2)
        ef4, mu4, skipConnect4, forwardMap4 = self.ec4(ef3, mu3)
        ef5, mu5, skipConnect5, forwardMap5 = self.ec5(ef4, mu4)
        ef6, mu6, skipConnect6, forwardMap6 = self.ec6(ef5, mu5)
        ef7, _, _, _ = self.ec7(ef6, mu6)


        reverseMap1, revMu1 = self.reverseConv1(1 - masks)
        reverseMap2, revMu2 = self.reverseConv2(revMu1)
        reverseMap3, revMu3 = self.reverseConv3(revMu2)
        reverseMap4, revMu4 = self.reverseConv4(revMu3)
        reverseMap5, revMu5 = self.reverseConv5(revMu4)
        reverseMap6, _ = self.reverseConv6(revMu5)

        concatMap6 = torch.cat((forwardMap6, reverseMap6), 1)
        dcFeatures1 = self.dc1(skipConnect6, ef7, concatMap6)

        concatMap5 = torch.cat((forwardMap5, reverseMap5), 1)
        dcFeatures2 = self.dc2(skipConnect5, dcFeatures1, concatMap5)

        concatMap4 = torch.cat((forwardMap4, reverseMap4), 1)
        dcFeatures3 = self.dc3(skipConnect4, dcFeatures2, concatMap4)

        concatMap3 = torch.cat((forwardMap3, reverseMap3), 1)
        dcFeatures4 = self.dc4(skipConnect3, dcFeatures3, concatMap3)

        concatMap2 = torch.cat((forwardMap2, reverseMap2), 1)
        dcFeatures5 = self.dc5(skipConnect2, dcFeatures4, concatMap2)

        concatMap1 = torch.cat((forwardMap1, reverseMap1), 1)
        dcFeatures6 = self.dc6(skipConnect1, dcFeatures5, concatMap1)

        dcFeatures7 = self.dc7(dcFeatures6)

        output = (self.tanh(dcFeatures7) + 1) / 2

        # pdb.set_trace()
        # (Pdb) pp output.size()
        # torch.Size([1, 3, 1024, 1024])

        return output


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__

        # pdb.set_trace()
        # (Pdb) pp init_type
        # 'gaussian'

        if (classname.find('Conv') == 0 or classname.find('Linear') == 0 ) and hasattr(m, 'weight'):
            if (init_type == 'gaussian'):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif (init_type == 'xavier'):
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif (init_type == 'kaiming'):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif (init_type == 'orthogonal'):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif (init_type == 'default'):
                pass
            else:
                assert 0, 'Unsupported initialization: {}'.format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class GaussActivation(nn.Module):
    def __init__(self, a, mu, sigma1, sigma2):
        super(GaussActivation, self).__init__()

        self.a = Parameter(torch.tensor(a, dtype=torch.float32))
        self.mu = Parameter(torch.tensor(mu, dtype=torch.float32))
        self.sigma1 = Parameter(torch.tensor(sigma1, dtype=torch.float32))
        self.sigma2 = Parameter(torch.tensor(sigma2, dtype=torch.float32))

        # pdb.set_trace()

    
    def forward(self, inputFeatures):

        # pdb.set_trace()

        self.a.data = torch.clamp(self.a.data, 1.01, 6.0)
        self.mu.data = torch.clamp(self.mu.data, 0.1, 3.0)
        self.sigma1.data = torch.clamp(self.sigma1.data, 0.5, 2.0)
        self.sigma2.data = torch.clamp(self.sigma2.data, 0.5, 2.0)

        lowerThanMu = inputFeatures < self.mu
        largerThanMu = inputFeatures >= self.mu

        # pdb.set_trace()

        leftValuesActiv = self.a * torch.exp(- self.sigma1 * ( (inputFeatures - self.mu) ** 2 ) )
        leftValuesActiv.masked_fill_(largerThanMu, 0.0)

        rightValueActiv = 1 + (self.a - 1) * torch.exp(- self.sigma2 * ( (inputFeatures - self.mu) ** 2 ) )
        rightValueActiv.masked_fill_(lowerThanMu, 0.0)

        output = leftValuesActiv + rightValueActiv

        # pdb.set_trace()

        return output

# mask updating functions, we recommand using alpha that is larger than 0 and lower than 1.0
class MaskUpdate(nn.Module):
    def __init__(self, alpha):
        super(MaskUpdate, self).__init__()

        self.updateFunc = nn.ReLU(True)
        #self.alpha = Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.alpha = alpha
    def forward(self, inputMaskMap):
        """ self.alpha.data = torch.clamp(self.alpha.data, 0.6, 0.8)
        print(self.alpha) """

        return torch.pow(self.updateFunc(inputMaskMap), self.alpha)

# learnable forward attention conv layer
class ForwardAttentionLayer(nn.Module):
    def __init__(self, inputChannels, outputChannels, kernelSize, stride, 
        padding, dilation=1, groups=1, bias=False):
        super(ForwardAttentionLayer, self).__init__()

        self.conv = nn.Conv2d(inputChannels, outputChannels, kernelSize, stride, padding, dilation, \
            groups, bias)

        if inputChannels == 4:
            self.maskConv = nn.Conv2d(3, outputChannels, kernelSize, stride, padding, dilation, \
                groups, bias)
        else:
            self.maskConv = nn.Conv2d(inputChannels, outputChannels, kernelSize, stride, padding, \
                dilation, groups, bias)
        
        self.conv.apply(weights_init())
        self.maskConv.apply(weights_init())

        self.activationFuncG_A = GaussActivation(1.1, 2.0, 1.0, 1.0)
        self.updateMask = MaskUpdate(0.8)


    def forward(self, inputFeatures, inputMasks):
        convFeatures = self.conv(inputFeatures)
        maskFeatures = self.maskConv(inputMasks)
        #convFeatures_skip = convFeatures.clone()

        maskActiv = self.activationFuncG_A(maskFeatures)
        convOut = convFeatures * maskActiv

        maskUpdate = self.updateMask(maskFeatures)

        return convOut, maskUpdate, convFeatures, maskActiv

# forward attention gather feature activation and batchnorm
class ForwardAttention(nn.Module):
    def __init__(self, inputChannels, outputChannels, bn=False, sample='down-4', \
        activ='leaky', convBias=False):
        super(ForwardAttention, self).__init__()

        if sample == 'down-4':
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 4, 2, 1, bias=convBias)
        elif sample == 'down-5':
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 5, 2, 2, bias=convBias)
        elif sample == 'down-7':
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 7, 2, 3, bias=convBias)
        elif sample == 'down-3':
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 3, 2, 1, bias=convBias)
        else:
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 3, 1, 1, bias=convBias)
        
        if bn:
            self.bn = nn.BatchNorm2d(outputChannels)
        
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
    
    def forward(self, inputFeatures, inputMasks):
        #pdb.set_trace()

        features, maskUpdated, convPreF, maskActiv = self.conv(inputFeatures, inputMasks)

        if hasattr(self, 'bn'):
            features = self.bn(features)
        if hasattr(self, 'activ'):
            features = self.activ(features)

        return features, maskUpdated, convPreF, maskActiv


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



class DiscriminatorDoubleColumn(nn.Module):
    def __init__(self, inputChannels):
        super(DiscriminatorDoubleColumn, self).__init__()

        self.globalConv = nn.Sequential(
            nn.Conv2d(inputChannels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2 , inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
        )

        self.localConv = nn.Sequential(
            nn.Conv2d(inputChannels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2 , inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # pdb.set_trace()
        
        self.fusionLayer = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=4),
            nn.Sigmoid()
        )

    def forward(self, batches, masks):
        globalFt = self.globalConv(batches * masks)
        localFt = self.localConv(batches * (1 - masks))

        concatFt = torch.cat((globalFt, localFt), 1)

        # pdb.set_trace()
        return self.fusionLayer(concatFt).view(batches.size()[0], -1)


def model_load(model, path):
    """Load model."""
    if not os.path.exists(path):
        print("Model '{}' does not exist.".format(path))
        return

    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)


def model_save(model, path):
    """Save model."""
    torch.save(model.state_dict(), path)

def model_export():
    """Export model to onnx."""

    import onnx
    from onnx import optimizer

    # xxxx--modify here
    onnx_file = "model.onnx"
    weight_file = "checkpoint/weight.pth"

    # 1. Load model
    print("Loading model ...")
    model = ImagePatchModel()
    model_load(model, weight_file)
    model.eval()

    # 2. Model export
    print("Export model ...")
    # xxxx--modify here
    dummy_input = torch.randn(1, 3, 512, 512)
    input_names = [ "input" ]
    output_names = [ "output" ]
    torch.onnx.export(model, dummy_input, onnx_file,
                    input_names=input_names, 
                    output_names=output_names,
                    verbose=True,
                    opset_version=11,
                    keep_initializers_as_inputs=True,
                    export_params=True)

    # 3. Optimize model
    print('Checking model ...')
    model = onnx.load(onnx_file)
    onnx.checker.check_model(model)

    print("Optimizing model ...")
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(model, passes)
    onnx.save(optimized_model, onnx_file)

    # 4. Visual model
    # python -c "import netron; netron.start('model.onnx')"


def get_model():
    """Create model."""
    model = ImagePatchModel(4, 3)
    return model


class Counter(object):
    """Class Counter."""

    def __init__(self):
        """Init average."""
        self.reset()

    def reset(self):
        """Reset average."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(loader, model, optimizer, device, tag=''):
    """Trainning model ..."""

    total_loss = Counter()

    model.train()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            predicts = model(images)

            # xxxx--modify here
            loss = nn.L1Loss(predicts, targets)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # Update loss
            total_loss.update(loss_value, count)

            t.set_postfix(loss='{:.6f}'.format(total_loss.avg))
            t.update(count)

            # Optimizer
            optimizer.zero_grad()
            if os.environ["ENABLE_APEX"] == "YES":
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        return total_loss.avg


def valid_epoch(loader, model, device, tag=''):
    """Validating model  ..."""

    valid_loss = Counter()

    model.eval()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            # Predict results without calculating gradients
            with torch.no_grad():
                predicts = model(images)

            # xxxx--modify here
            valid_loss.update(loss_value, count)
            t.set_postfix(loss='{:.6f}'.format(valid_loss.avg))
            t.update(count)


def model_setenv():
    """Setup environ  ..."""

    # random init ...
    import random
    random.seed(42)
    torch.manual_seed(42)

    # Set default environment variables to avoid exceptions
    if os.environ.get("ONLY_USE_CPU") != "YES" and os.environ.get("ONLY_USE_CPU") != "NO":
        os.environ["ONLY_USE_CPU"] = "NO"

    if os.environ.get("ENABLE_APEX") != "YES" and os.environ.get("ENABLE_APEX") != "NO":
        os.environ["ENABLE_APEX"] = "YES"

    if os.environ.get("DEVICE") != "YES" and os.environ.get("DEVICE") != "NO":
        os.environ["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Is there GPU ?
    if not torch.cuda.is_available():
        os.environ["ONLY_USE_CPU"] = "YES"

    # export ONLY_USE_CPU=YES ?
    if os.environ.get("ONLY_USE_CPU") == "YES":
        os.environ["ENABLE_APEX"] = "NO"
    else:
        try:
            from apex import amp
        except:
            os.environ["ENABLE_APEX"] = "NO"

    # Running on GPU if available
    if os.environ.get("ONLY_USE_CPU") == "YES":
        os.environ["DEVICE"] = 'cpu'
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])
    print("  ONLY_USE_CPU: ", os.environ["ONLY_USE_CPU"])
    print("  ENABLE_APEX: ", os.environ["ENABLE_APEX"])


def infer_perform():
    """Model infer performance ..."""

    model_setenv()
    device = os.environ["DEVICE"]

    model = ImagePatchModel()
    model.eval()
    model = model.to(device)

    if os.environ["ENABLE_APEX"] == "YES":
        from apex import amp
        model = amp.initialize(model, opt_level="O1")
    print(model)

    # for i in tqdm(range(10)):
    #     input = torch.randn(2, 3, 512, 512)
    #     input = input.to(device)

    #     with torch.no_grad():
    #         output = model(input)

    #     del input, output

if __name__ == '__main__':
    """Test model ..."""

    # model_export()
    infer_perform()
