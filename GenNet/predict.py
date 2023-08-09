import os
import time

import cmapy
import cv2
from PIL import Image

import torch
import torchvision.transforms as T

from utils import misc
# from networks import AESwin as AE
from networks import AEViT as AE
from my_dataset import Dataset, DataTransform



def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    weights_path = "/home/long/fine-tune/PlanningNet/2nd_aeswin_320k_path_patch_C1/model_0.pth"
    weights_path = "/home/long/fine-tune/PPNet/GenNet/GenNet_c3_emb24.pth"
    data_path = "/home/long/dataset/planning224_1_seg_patch/result_ompl/map_cropped_trans.png"
    data_root = "/home/long/dataset/exp_diff_data_generation/Data_RRTstar_40*250"
    data_root = "/home/long/dataset/exp_diff_data_generation"
    data_path = data_root + "/result_space_c3"
    # data_path = "/home/long/dataset/figure/space2471.png"
    result_path = "/home/long/dataset/figure"# + '/result'
    result_path = data_root + "/result_path_c3"
    img_channels = 1
    out_channels = 1
    img_resolution = 224
    embedding_dim = 24
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(data_path), f"data {data_path} not found."
    assert os.path.exists(result_path), f"result {result_path} not found."

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 创建模型 打印参数
    model = AE(img_channels=img_channels, out_channels=out_channels, img_resolution=img_resolution, dim=embedding_dim)
    img_in = torch.empty([1, img_channels, img_resolution, img_resolution], device=device)
    misc.print_module_summary(model.to(device), img_in)

    # load weights
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    model.load_state_dict(weights_dict)
    model.to(device)


    # load image
    if data_path.split('/')[-1].split('.')[-1] != 'png':
        test_dataset = Dataset(data_path, transforms=DataTransform(), subset='test')
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  num_workers=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  collate_fn=test_dataset.collate_fn)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        inference_time = []
        if data_path.split('/')[-1].split('.')[-1] == 'png':
            img = T.ToTensor()(Image.open(data_path))*255
            output = model(img.unsqueeze(dim=0).to(device)).squeeze()
            # post norm
            ma = torch.max(output)
            mi = torch.min(output)
            output = (output - mi) / (ma - mi)
            output_heatmap = T.ToPILImage()(output.squeeze()).convert('L')
            output = T.ToPILImage()(output.squeeze()).convert('L')
            img_name = "./{}.png".format(data_path.split('/')[-1].split('.')[0])
            output.save(img_name)
            # output_heatmap.save(img_name)
            # output_heatmap = cv2.imread(img_name)
            # output_heatmap = Image.fromarray(cv2.applyColorMap(output_heatmap, cmapy.cmap('jet_r')))
            # output_heatmap.save(img_name)
            return
        for index, [image, target] in enumerate(test_loader):
            image, target = image.to(device), target.to(device)
            t_start = time_synchronized()
            output = model(image).squeeze()
            t_end = time_synchronized()
            if index >= 1000:  # warm up
                inference_time.append(t_end - t_start)
            if index%100 == 0:
                print('index', index)
            # post norm
            ma = torch.max(output)
            mi = torch.min(output)
            output = (output - mi) / (ma - mi)
            output_heatmap = T.ToPILImage()(output.squeeze()).convert('L')
            output = T.ToPILImage()(output.squeeze()).convert('L')
            img_name = "{}/{}.png".format(result_path, test_dataset.images[index].split('/')[-1].split('.')[0])
            # # save
            output.save(img_name)
            # # convert to heatmap
            # output_heatmap.save(img_name)
            # output_heatmap = cv2.imread(img_name)
            # output_heatmap = Image.fromarray(cv2.applyColorMap(output_heatmap, cmapy.cmap('jet_r')))
            # output_heatmap.save(img_name)

        print("inference time: {} per sample".format(sum(inference_time[-100:]) / 100))


if __name__ == '__main__':
    main()
