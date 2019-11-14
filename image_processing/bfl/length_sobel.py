import cv2
import numpy as np
import os
import envision

def showimage(imagename,image):
	cv2.imshow(imagename,image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def head_pixel_value(image):

	img = image.copy()


	kernel = np.ones((5,5),np.float32)/25
	dst = cv2.filter2D(img,-1,kernel)
	showimage("blur",dst)
	# img = cv2.medianblur

if __name__ == "__main__":
	path = "rpi-7_images/big_shaft_15/"

	images=[]

	for r,d,f in os.walk(path):
		for files in f:
			if ".jpg" in files:
				images.append(os.path.join(r,files))

	for img in images:


		original_image = cv2.imread(img,0)

		showimage(img,original_image)

		head_pixel_value(original_image)

		break


