import tkinter
from tkinter import filedialog, dialog
import torch
from detect_plate import *
import torchvision.transforms as transforms

from torch.utils.data import  DataLoader,Dataset

import matplotlib.pyplot as plt

import torch.nn as nn
import os
from PIL import Image, ImageTk

from utils.DealDataset import DealDataset

# ...
from PIL import Image, ImageTk


def image_resize(img, screen_width=1000, screen_height=500):
    image = img

    raw_width, raw_height = image.size[0], image.size[1]
    max_width, max_height = raw_width, screen_height
    min_width = max(raw_width, max_width)
    # 按照比例缩放
    min_height = int(raw_height * min_width / raw_width)
    # 第1次快速调整
    while min_height > screen_height:
        min_height = int(min_height * .9533)
    # 第2次精确微调
    while min_height < screen_height:
        min_height += 1
    # 按照比例缩放
    min_width = int(raw_width * min_height / raw_height)
    # 适应性调整
    while min_width > screen_width:
        min_width -= 1
    # 按照比例缩放
    min_height = int(raw_height * min_width / raw_width)
    return image.resize((min_width, min_height))

def open_file_output():
    '''
    打开文件
    :return:local_
    '''
    global file_path
    global file_text
    global photo
    global img
    file_path = filedialog.askopenfilename(title=u'选择文件')
    print('打开文件：', file_path)
    if file_path is not None:
        file_text = "文件路径为：" + file_path


    img = Image.open(file_path)  # 打开图片
    img = image_resize(img)

    photo = ImageTk.PhotoImage(img)  # 用PIL模块的PhotoImage打开

    imglabel = tkinter.Label(window,bd = 10, image=photo)
    imglabel.place(relx=0, rely=0)




def run():

    global output_text
    global output_color
    global ori_img
    global save_img_path

    img = cv_imread(file_path)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, opt.img_size,
                                         is_color=opt.is_color)

    output_text = dict_list[0]['plate_no']
    output_color = dict_list[0]['plate_color']

    ori_img = draw_result(img, dict_list)

    # img_output = Image.open(ori_img)
    img_output = Image.fromarray(np.uint8(ori_img))
    img_output = image_resize(img_output)

    img_output = ImageTk.PhotoImage(img_output)  # 用PIL模块的PhotoImage打开

    imglabel2 = tkinter.Label(window, bd=10, image=img_output)
    imglabel2.image = img_output  # 更新时防止一闪而过代码（非常重要）
    imglabel2.place(relx=0, rely=0)


    img_name = os.path.basename(opt.image_path)
    save_img_path = os.path.join(save_path, file_path.split('/')[-1])
    # cv2.imwrite(save_img_path, ori_img)

    # capture.release()
    # out.release()
    # cv2.destroyAllWindows()

    t1 = output_text
    text1 = tkinter.Label(window, bd=10, font = 40,fg='red', bg='white', text=t1)
    text1.place(relx=0.52, rely=0.8)  # 相对位置，放置文本

    t2 = output_color
    text2 = tkinter.Label(window, bd=10, font=40, fg='red', bg='white', text=t2)
    text2.place(relx=0.47, rely=0.8)  # 相对位置，放置文本

def load_model_2():
    text1 = tkinter.Label(window, bd=10, font=40,fg='red', text = "模型加载成功")

    text1.place(relx=0.27, rely=0.8)

def save_img():
    cv2.imwrite(save_img_path, ori_img) #  save_img_path  写的复杂（变量传递太多），可以简化的
    text5 = tkinter.Label(window, bd=10, fg='red',font=40, text="保存成功")

    text5.place(relx=0.68, rely=0.8)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default='weights/plate_detect.pt',
                        help='model.pt path(s)')  # 检测模型
    parser.add_argument('--rec_model', type=str, default='weights/plate_rec_color.pth',
                        help='model.pt path(s)')  # 车牌识别+颜色识别模型
    parser.add_argument('--is_color', type=bool, default=True, help='plate color')  # 是否识别颜色
    parser.add_argument('--image_path', type=str, default='imgs', help='source')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--output', type=str, default='result1', help='source')
    parser.add_argument('--video', type=str, default='', help='source')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    opt = parser.parse_args()
    print(opt)
    save_path = opt.output
    count = 0
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    detect_model = load_model(opt.detect_model, device)  # 初始化检测模型
    plate_rec_model = init_model(device, opt.rec_model, is_color=opt.is_color)  # 初始化识别模型
    # 算参数量
    total = sum(p.numel() for p in detect_model.parameters())
    total_1 = sum(p.numel() for p in plate_rec_model.parameters())
    # print("detect params: %.2fM,rec params: %.2fM" % (total / 1e6, total_1 / 1e6))

    # plate_color_model =init_color_model(opt.color_model,device)
    time_all = 0
    time_begin = time.time()


    # print(f"all frame is {frame_count},average fps is {fps_all / frame_count} fps")



    window = tkinter.Tk()
    window.title('车牌识别系统')
    window.geometry('1000x800')

    button1 = tkinter.Button(window, text='选择车牌', command=open_file_output, width=10, height=2)  # 加括号会自动执行（！！）
    button5 = tkinter.Button(window, text='退出', bg="red", fg="white", command=lambda: window.destroy(), width=10,
                             height=2)
    button3 = tkinter.Button(window, text='处理', command=run, width=10, height=2)  # 加括号会自动执行（！！）
    button2 = tkinter.Button(window, text='加载模型', command=load_model_2, width=10, height=2)
    button4 = tkinter.Button(window, text='结果保存', command=save_img, width=10, height=2)

    button1.place(relx=0.17, rely=0.8, anchor='se')  # 相对位置，放置按钮
    button2.place(relx=0.37, rely=0.8, anchor='se')  # 相对位置，放置按钮
    button3.place(relx=0.57, rely=0.8, anchor='se')  # 相对位置，放置按钮
    button4.place(relx=0.77, rely=0.8, anchor='se')  # 相对位置，放置按钮
    button5.place(relx=0.97, rely=0.8, anchor='se')  # 相对位置，放置按钮

window.mainloop()

