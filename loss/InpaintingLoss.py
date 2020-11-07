import torch
from torch import nn
from torch import autograd
from tensorboardX import SummaryWriter
from models.net_D import DiscriminatorDoubleColumn

import pdb

# modified from WGAN-GP
def gradient_penalty(netD, real_data, fake_data, masks):
    BATCH_SIZE = real_data.size()[0]
    DIM = real_data.size()[2]
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 3, DIM, DIM)
    alpha = alpha.cuda()
    
    fake_data = fake_data.view(BATCH_SIZE, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates, masks)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    Lambda = 10.0                              
    gradient_penalty_x = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda

    # pdb.set_trace()

    return gradient_penalty_x.sum().mean()


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    pdb.set_trace()

    return gram


#tv loss
# def total_variation_loss(image):
#     # shift one pixel and get difference (for both x and y direction)
#     loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
#         torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))

#     # pdb.set_trace()

#     return loss

#VGG16 feature extract
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load('./vgg16-397923af.pth'))
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

        pdb.set_trace()

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))

        pdb.set_trace()

        return results[1:]


class InpaintingLossWithGAN(nn.Module):
    def __init__(self, lr, betasInit=(0.5, 0.9)):
        # Lamda=10.0, lr = 0.00001
        super(InpaintingLossWithGAN, self).__init__()
        self.l1 = nn.L1Loss()
        self.extractor = VGG16FeatureExtractor()
        self.net_D = DiscriminatorDoubleColumn(3)
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=lr, betas=betasInit)


    def forward(self, input, mask, output, gt):
        self.net_D.zero_grad()
        D_real = self.net_D(gt, mask)
        D_real = D_real.mean().sum() * -1
        D_fake = self.net_D(output, mask)
        D_fake = D_fake.mean().sum() * 1
        gp = gradient_penalty(self.net_D, gt, output, mask)
        D_loss = D_fake - D_real + gp

        # netD optimize
        self.optimizer_D.zero_grad()
        D_loss.backward(retain_graph=True)
        self.optimizer_D.step()

        output_comp = mask * input + (1 - mask) * output

        holeLoss = 6 * self.l1((1 - mask) * output, (1 - mask) * gt)
        validAreaLoss = self.l1(mask * output, mask * gt)   

        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
            feat_output = self.extractor(torch.cat([output]*3, 1))
            feat_gt = self.extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('only gray an')

        prcLoss = 0.0
        for i in range(3):
            prcLoss += 0.01 * self.l1(feat_output[i], feat_gt[i])
            prcLoss += 0.01 * self.l1(feat_output_comp[i], feat_gt[i])

        styleLoss = 0.0
        for i in range(3):
            styleLoss += 120 * self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            styleLoss += 120 * self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))
        
        GLoss = holeLoss + validAreaLoss + prcLoss + styleLoss + 0.1 * D_fake

        return GLoss.sum()
