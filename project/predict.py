"""Model predict."""
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
import glob
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import get_model, model_load, model_setenv
from data import image_with_mask
from tqdm import tqdm
import pdb

if __name__ == "__main__":
    """Predict."""

    model_setenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="models/ImagePatch.pth", help="checkpint file")
    parser.add_argument('--input', type=str, default="dataset/predict/image/*.png", help="input image")
    args = parser.parse_args()

    # CPU or GPU ?
    device = torch.device(os.environ["DEVICE"])

    model = get_model()
    model_load(model, args.checkpoint)
    model.to(device)
    model.eval()

    if os.environ["ENABLE_APEX"] == "YES":
        from apex import amp
        model = amp.initialize(model, opt_level="O1")

    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    image_filenames = glob.glob(args.input)
    progress_bar = tqdm(total = len(image_filenames))

    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        # image
        image = Image.open(filename).convert("RGB")
        input_tensor = totensor(image).unsqueeze(0).to(device)

        # mask
        mask_filename = os.path.dirname(os.path.dirname(filename)) \
            + "/mask/" + os.path.basename(filename)
        mask_image = Image.open(mask_filename).convert("RGB")
        mask_tensor = totensor(mask_image).unsqueeze(0).to(device)

        new_input_tensor, new_mask_tensor = image_with_mask(input_tensor, mask_tensor)
        with torch.no_grad():
            output_tensor = model(new_input_tensor, new_mask_tensor)

        output_tensor = output_tensor.clamp(0, 1.0).squeeze()
        output_filename = os.path.dirname(os.path.dirname(filename)) \
            + "/output/" + os.path.basename(filename)
        toimage(output_tensor.cpu()).save(output_filename)
