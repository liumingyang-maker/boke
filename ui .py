import tkinter
from tkinter import filedialog
from PIL import Image, ImageTk
from detect_plate import *

# Set device to use
device = torch.device("cpu")


# Resize the image while maintaining its aspect ratio
def image_resize(img, screen_width=1000, screen_height=500):
    # Get the original image dimensions
    raw_width, raw_height = img.size[0], img.size[1]
    # Set the maximum width and height based on the screen dimensions
    max_width, max_height = screen_width, screen_height

    # Scale the image to fit within the maximum height
    scaled_height = min(raw_height, max_height)
    scaled_width = int(scaled_height * raw_width / raw_height)

    # If the scaled width exceeds the maximum width, scale the image down
    while scaled_width > max_width:
        scaled_width -= 1
        scaled_height = int(scaled_width * raw_height / raw_width)

    # If the scaled height is less than the maximum height, scale the image up
    while scaled_height < max_height:
        scaled_height += 1
        scaled_width = int(scaled_height * raw_width / raw_height)

    # Resize the image to the final dimensions
    resized_image = img.resize((scaled_width, scaled_height))

    return resized_image


# Open file and output image
def open_file_output():
    """
    Open a file and display it in the GUI.

    Global variables:
        file_path: The path of the opened file
        img: The PIL Image object
        photo: The PhotoImage object
    """
    global file_path
    global file_text
    global photo
    global img

    file_path = filedialog.askopenfilename(title=u'Select File')
    print('Open File：', file_path)
    if not file_path:
        return
    # Open the file dialog
    file_text = "The file path is：" + file_path
    img = Image.open(file_path)

    try:
        img = image_resize(img)
    except Exception as e:
        print(f"Error while resizing image: {e}")
        return

    photo = ImageTk.PhotoImage(img)
    imglabel = tkinter.Label(window, bd=10, image=photo)
    imglabel.place(relx=0, rely=0)


# Run the License Plate Recognition process
def run():
    """
    Run the License Plate Recognition process on the input image.

    Global variables:
        output_text: Text of the detected license plate number
        output_color: Color of the detected license plate
        ori_img: Original image with recognized license plate and plate color
        save_img_path: Path to save the output image
    """
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

    img_output = Image.fromarray(np.uint8(ori_img))
    img_output = image_resize(img_output)
    img_output = ImageTk.PhotoImage(img_output)

    imglabel2 = tkinter.Label(window, bd=10, image=img_output)
    imglabel2.image = img_output

    imglabel2.place(relx=0, rely=0)

    img_name = os.path.basename(opt.image_path)
    save_img_path = os.path.join(save_path, file_path.split('/')[-1])

    display_text()


# Show a label to indicate successful model loading
def load_model_2():
    # Create a label widget with the specified text, font, and color
    text1 = tkinter.Label(window, bd=10, font=40, fg='red', text="Model loading success")
    # Position the label at the specified coordinates
    text1.place(relx=0.27, rely=0.8)

# Save the processed image
def save_img():
    # Save the processed image (ori_img) to the specified path (save_img_path)
    cv2.imwrite(save_img_path, ori_img)
    # Create a label widget to show that the image was saved successfully
    text5 = tkinter.Label(window, bd=10, fg='red', font=40, text="save successfully")
    # Position the label at the specified coordinates
    text5.place(relx=0.68, rely=0.8)

# Display the detected license plate number and color
def display_text():
    # Store the detected license plate number in t1
    t1 = output_text
    # Create a label widget to display the detected license plate number with the specified text, font, and color
    text1 = tkinter.Label(window, bd=10, font=40, fg='red', bg='white', text=t1)
    # Position the label at the specified coordinates
    text1.place(relx=0.52, rely=0.8)

    # Store the detected license plate color in t2
    t2 = output_color
    # Create a label widget to display the detected license plate color with the specified text, font, and color
    text2 = tkinter.Label(window, bd=10, font=40, fg='red', bg='white', text=t2)
    # Position the label at the specified coordinates
    text2.place(relx=0.47, rely=0.8)

# Main function
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default='weights/plate_detect.pt',
                        help='model.pt path(s)')
    parser.add_argument('--rec_model', type=str, default='weights/plate_rec_color.pth',
                        help='model.pt path(s)')
    parser.add_argument('--is_color', type=bool, default=True, help='plate color')
    parser.add_argument('--image_path', type=str, default='imgs', help='source')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--output', type=str, default='result1', help='source')
    parser.add_argument('--video', type=str, default='', help='source')

    # Store the parsed arguments in the opt variable
    opt = parser.parse_args()
    # Print the parsed arguments
    print(opt)
    # Set the save path for the processed images
    save_path = opt.output
    # Create the save directory if it does not exist
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Load the detection model
    detect_model = load_model(opt.detect_model, device)
    # Initialize the license plate recognition model
    plate_rec_model = init_model(device, opt.rec_model, is_color=opt.is_color)

    # Create the main GUI window
    window = tkinter.Tk()
    window.title('Vehicle License Plate Recognition System')
    window.geometry('1000x800')

    # Create buttons for various functions
    button1 = tkinter.Button(window, text='Select', command=open_file_output, width=10, height=2)
    button5 = tkinter.Button(window, text='EXIT', bg="red", fg="white", command=lambda: window.destroy(), width=10,
                             height=2)
    button3 = tkinter.Button(window, text='Process', command=run, width=10, height=2)
    button2 = tkinter.Button(window, text='Load Model', command=load_model_2, width=10, height=2)
    button4 = tkinter.Button(window, text='Save', command=save_img, width=10, height=2)

    button1.place(relx=0.17, rely=0.8, anchor='se')
    button2.place(relx=0.37, rely=0.8, anchor='se')
    button3.place(relx=0.57, rely=0.8, anchor='se')
    button4.place(relx=0.77, rely=0.8, anchor='se')
    button5.place(relx=0.97, rely=0.8, anchor='se')

    window.mainloop()
