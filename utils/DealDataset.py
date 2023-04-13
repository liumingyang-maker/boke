from torch.utils.data import  DataLoader,Dataset
import os
from PIL import Image

class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self,transforms ,datapath = None):
        self.transforms = transforms
        self.datapath = datapath
        self.img_path = os.listdir(self.datapath)

    def __getitem__(self, index):
        path_temp = self.datapath + '\\'  + self.img_path[index]

        data = Image.open(path_temp).convert('RGB')

        data = self.transforms(data)

        if self.img_path[index][1] =='M':
            label = int(self.img_path[index][0])
        else:
            label = int(self.img_path[index][0])+10

        return data, label

    def __len__(self):
        return len(self.img_path)

    def get_name(self,index):
        return self.img_path[index]

    def get_img_path(self,index):
        return self.datapath + '\\'  + self.img_path[index]