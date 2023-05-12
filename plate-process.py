import os
import shutil
import cv2
import numpy as np
# Traverse all the files under the given root directory and add the paths of all jpg images to allFIleList
def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            if temp.endswith(".jpg"):
                allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)
# Sort the given four points and return a new array containing points ordered in a specific sequence
# The order of the points is: [top-left, top-right, bottom-right, bottom-left]
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    pts=pts[:4,:]
    rect = np.zeros((5, 2), dtype = "float32")
 
    # Sort by the difference between the points, finding the smallest difference (top-right) and the largest difference (bottom-left)
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Sort by the difference between the points, finding the smallest difference (top-right) and the largest difference (bottom-left)
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return rect
# Extract a part of the CCPD dataset and move some of the images to a new folder
def get_partical_ccpd():
    ccpd_dir = r"/mnt/Gpan/BaiduNetdiskDownload/CCPD1/CCPD2020/ccpd_green"
    save_Path = r"ccpd/green_plate"
    folder_list = os.listdir(ccpd_dir)
    for folder_name in folder_list:
        count=0
        folder_path = os.path.join(ccpd_dir,folder_name)
        if os.path.isfile(folder_path):
            continue
        if folder_name == "ccpd_fn":
            continue
        name_list = os.listdir(folder_path)
        
        save_folder=save_Path
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        for name in name_list:
            file_path = os.path.join(folder_path,name)
            count+=1
            if count>1000:
                break
            new_file_path =os.path.join(save_folder,name)
            shutil.move(file_path,new_file_path)
            print(count,new_file_path)
# Parse the bounding box of the license plate and the coordinates of the keypoints from the given image path
def get_rect_and_landmarks(img_path):
   file_name = img_path.split("/")[-1].split("-")
   landmarks_np =np.zeros((5,2))
   # Extract the bounding box and landmark information from the file name
   rect = file_name[2].split("_")
   landmarks=file_name[3].split("_")
   rect_str = "&".join(rect)
   landmarks_str= "&".join(landmarks)
   rect= rect_str.split("&")
   landmarks=landmarks_str.split("&")
   rect=[int(x) for x in rect]
   landmarks=[int(x) for x in landmarks]
   # Assign the landmark points to the landmarks_np array
   for i in range(4):
        landmarks_np[i][0]=landmarks[2*i]
        landmarks_np[i][1]=landmarks[2*i+1]
   # Calculate the middle point between the two last keypoints and add it to the landmarks array
   middle_landmark_w =int((landmarks[4]+landmarks[6])/2) 
   middle_landmark_h =int((landmarks[5]+landmarks[7])/2) 
   landmarks.append(middle_landmark_w)
   landmarks.append(middle_landmark_h)
   # Reorder the points in the landmarks_np array
   landmarks_np_new=order_points(landmarks_np)
   landmarks_np_new[4]=np.array([middle_landmark_w,middle_landmark_h])
   return rect,landmarks,landmarks_np_new
# Convert x1, x2, y1, y2 to YOLO format
def x1x2y1y2_yolo(rect,landmarks,img):
    h,w,c =img.shape
    # Ensure the bounding box coordinates are within the image dimensions
    rect[0] = max(0, rect[0])
    rect[1] = max(0, rect[1])
    rect[2] = min(w - 1, rect[2]-rect[0])
    rect[3] = min(h - 1, rect[3]-rect[1])
    # Initialize an empty annotation array
    annotation = np.zeros((1, 14))
    annotation[0, 0] = (rect[0] + rect[2] / 2) / w  # cx
    annotation[0, 1] = (rect[1] + rect[3] / 2) / h  # cy
    annotation[0, 2] = rect[2] / w  # w
    annotation[0, 3] = rect[3] / h  # h

    annotation[0, 4] = landmarks[0] / w  # l0_x
    annotation[0, 5] = landmarks[1] / h  # l0_y
    annotation[0, 6] = landmarks[2] / w  # l1_x
    annotation[0, 7] = landmarks[3] / h  # l1_y
    annotation[0, 8] = landmarks[4] / w  # l2_x
    annotation[0, 9] = landmarks[5] / h # l2_y
    annotation[0, 10] = landmarks[6] / w  # l3_x
    annotation[0, 11] = landmarks[7] / h  # l3_y
    annotation[0, 12] = landmarks[8] / w  # l4_x
    annotation[0, 13] = landmarks[9] / h  # l4_y
    # Assign the landmark coordinates to the annotation array
    return annotation
# Convert the bounding box and landmarks (sorted) in XYWH format to YOLO format
def xywh2yolo(rect,landmarks_sort,img):
    h,w,c =img.shape
    # Ensure the bounding box coordinates are within the image dimensions
    rect[0] = max(0, rect[0])
    rect[1] = max(0, rect[1])
    rect[2] = min(w - 1, rect[2]-rect[0])
    rect[3] = min(h - 1, rect[3]-rect[1])
    # Initialize an empty annotation array
    annotation = np.zeros((1, 14))
    annotation[0, 0] = (rect[0] + rect[2] / 2) / w  # cx
    annotation[0, 1] = (rect[1] + rect[3] / 2) / h  # cy
    annotation[0, 2] = rect[2] / w  # w
    annotation[0, 3] = rect[3] / h  # h

    annotation[0, 4] = landmarks_sort[0][0] / w  # l0_x
    annotation[0, 5] = landmarks_sort[0][1] / h  # l0_y
    annotation[0, 6] = landmarks_sort[1][0] / w  # l1_x
    annotation[0, 7] = landmarks_sort[1][1] / h  # l1_y
    annotation[0, 8] = landmarks_sort[2][0] / w  # l2_x
    annotation[0, 9] = landmarks_sort[2][1] / h # l2_y
    annotation[0, 10] = landmarks_sort[3][0] / w  # l3_x
    annotation[0, 11] = landmarks_sort[3][1] / h  # l3_y
    annotation[0, 12] = landmarks_sort[4][0] / w  # l4_x
    annotation[0, 13] = landmarks_sort[4][1] / h  # l4_y
    return annotation
# Convert YOLO format back to x1, y1, x2, y2 and landmark coordinates
def yolo2x1y1x2y2(annotation,img):
    h,w,c = img.shape
    # Extract the bounding box and landmark information from the annotation array
    rect= annotation[:,0:4].squeeze().tolist()
    landmarks=annotation[:,4:].squeeze().tolist()
    # Calculate the actual bounding box coordinates in the image
    rect_w = w*rect[2]
    rect_h =h*rect[3]
    rect_x =int(rect[0]*w-rect_w/2)
    rect_y = int(rect[1]*h-rect_h/2)
    new_rect=[rect_x,rect_y,rect_x+rect_w,rect_y+rect_h]
    # Calculate the actual landmark coordinates in the image
    for i in range(5):
        landmarks[2*i]=landmarks[2*i]*w
        landmarks[2*i+1]=landmarks[2*i+1]*h
    return new_rect,landmarks
# Function to write the label file
def write_lable(file_path):
    pass


if __name__ == '__main__':
   file_root = r"ccpd/val"
   file_list=[]
   count=0
   # Recursively find all .jpg files in the directory
   allFilePath(file_root,file_list)
   for img_path in file_list:
        count+=1
        # img_path = r"ccpd_yolo_test/02-90_85-173&466_452&541-452&553_176&556_178&463_454&460-0_0_6_26_15_26_32-68-53.jpg"
        text_path= img_path.replace(".jpg",".txt")
        img =cv2.imread(img_path)
        # Get the bounding box and landmark information from the image path
        rect,landmarks,landmarks_sort=get_rect_and_landmarks(img_path)
        # annotation=x1x2y1y2_yolo(rect,landmarks,img)
        annotation=xywh2yolo(rect,landmarks_sort,img)
        str_label = "0 "
        for i in range(len(annotation[0])):
                str_label = str_label + " " + str(annotation[0][i])
        str_label = str_label.replace('[', '').replace(']', '')
        str_label = str_label.replace(',', '') + '\n'
        # Write the label string to a file
        with open(text_path,"w") as f:
                f.write(str_label)
        print(count,img_path)

        
        