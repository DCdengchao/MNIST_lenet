from LeNet5 import LeNet5
import torch
from readMnist import *
from myData import Mnist
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os

test_images = load_test_images()
test_labels = load_test_labels()

testData = Mnist(test_images, test_labels)
test_data = DataLoader(dataset=testData, batch_size=1, shuffle=True)

lenet5 = LeNet5()
lenet5.load_state_dict(torch.load('lenet5.pth', map_location='cpu'))
lenet5.eval()

# 创建保存图片的文件夹
save_dir = "mnist_result"
os.makedirs(save_dir, exist_ok=True)

showimg = False
js = 0

# 只保存前 20 张图片，防止生成太多文件
save_num = 20

with torch.no_grad():
    for i, (img, id) in enumerate(test_data):

        img = img.float()
        outid = lenet5(img)

        oid = torch.argmax(outid, dim=1)

        if oid.item() == id.item():
            js = js + 1

        if showimg == True and i < save_num:
            show_img = img.numpy()
            show_img = np.squeeze(show_img)

            true_id = id.item()
            pred_id = oid.item()

            maxv = np.max(show_img)
            minv = np.min(show_img)

            show_img = (show_img - minv) / (maxv - minv)
            show_img = (show_img * 255).astype(np.uint8)

            filename = f"{save_dir}/img_{i}_pred_{pred_id}_true_{true_id}.png"
            cv2.imwrite(filename, show_img)

            print(f"已保存：{filename}")

print('准确率：{:.6f}'.format(js / (i + 1)))