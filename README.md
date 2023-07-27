# Yolo-Entity-Detection
This project uses a YoLo pertained model to detect cars in images.

# Load YoLo Model from yolo.h5
We can download this h5 file from https://drive.google.com/file/d/1v-V94VX2JWIrsDN4u8t9QUCMfzS6x08c/view

# how to start
1. Pick an image file from the images folder, and put the image file path in carDetection.py
2. This system only works well for image sizes 1280 * 720. So if the image size is different, we need to run the image_resize function in carDetection.py
3. The YoLo model seems can't recognize entities well if the image is rotated. So we need to run image_rotate function to rotate back for these kind of images.
4. Run predict function to generate output images in "out" folder, You can find we highlight the cars we detected in these output images.

# Next steps
1. re-write the "draw_boxes" function in util.py to make this system work for different image sizes. We don't need to resize input images anymore.
2. Use the transfer learning technique to upgrade yolo model and improve the performance when the handling rotated images
