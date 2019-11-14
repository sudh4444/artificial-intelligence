import cv2
import numpy as np
from envision.convolution import convolve_sobel
import time
import os


kernel_list = []
debug = True
# debug = False
stack_img = []

def showimage(image_name,image):
	cv2.imshow(image_name,image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def custom_soften(image):

	# edges = cv2.Canny(img,100,200)
	# if debug:
	# 	showimage("canny",ed)

	# kernel = np.array((
	# 				[1,1,1,1,1],
	# 				[1,2,2,2,1],
	# 				[1,2,3,2,1],
	# 				[1,2,2,2,1],
	# 				[1,1,1,1,1]),dtype = "int")

	kernel = np.array((
				[1,1,1],
				[1,2,1],
				[1,1,1]),dtype = "int")
	kernel = kernel/9
	blur = cv2.filter2D(image,cv2.CV_8UC3,kernel)

	if debug:
		showimage("custom soften",blur)

	kernel = np.ones((7,7),np.uint8)
	erosion = cv2.erode(blur,kernel,iterations = 2)

	if debug:
		showimage("erosion",erosion)

	return erosion


def test_for_lap(image):
	global stack_img

	image = cv2.medianBlur(image,5)
	# cv2.imshow("blur",image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# if debug:
	# 	showimage("blur",blur)

	sobel = convolve_sobel(img=image,
                   threshold=30,
                   sobel_kernel_left_right=True,
                   sobel_kernel_right_left=True,
                   sobel_kernel_top_bottom=False,
                   sobel_kernel_bottom_top=False,
                   sobel_kernel_diagonal_top_left=False,
                   sobel_kernel_diagonal_bottom_left=False,
                   sobel_kernel_diagonal_top_right=False,
                   sobel_kernel_diagonal_bottom_right=False)
	sobel = cv2.convertScaleAbs(sobel)

	if debug:
		showimage("sobel",sobel)


	gaus = cv2.GaussianBlur(sobel, (3,3), -1)

	if debug:
		showimage("Gaussian",gaus)

	_,thresh = cv2.threshold(gaus,200,255,cv2.THRESH_BINARY)

	if debug:
		showimage("thresh",thresh)

	stack_img = cv2.bitwise_or(stack_img,thresh)

	if debug:
		showimage("stack image",stack_img)
		
	



def dynamic_crop(g_bk):
	# global stack_img

	g9 = g_bk.copy()

	g9 = cv2.medianBlur(g9,59)

	if debug:
		showimage("blur",g9)

	_,g9 = cv2.threshold(g9,10,255,cv2.THRESH_BINARY)

	if debug:
		showimage("thresh",g9)

	first_pix_loc1_x = 0
	y1 = 220  + 20
	y2 = 520  - 50-20
	x_fixed = 350
	
	mid_y = int(round(g9.shape[0]/2))
	for j1 in range(0, g9.shape[1]):
		if g9[mid_y][j1] ==255 and first_pix_loc1_x==0:
			first_pix_loc1_x = j1
			# print ("first pix", j1)
		g9[mid_y][j1] = 255

	for y in range(0,g9.shape[0]):
		if g9[y][x_fixed] ==255:
			y1 = y
			# print("h1",y1)
			break
		g9[y][x_fixed] = 255

	for y in reversed(range(g9.shape[0])):
		if g9[y][x_fixed] ==255:
			y2 = y
			# print("h1",y1)
			break
		g9[y][x_fixed] = 255


	if debug:
		showimage('dimensions', g9)
		# cv2.waitKey(0)



	width = g_bk.shape[1]
		# print("width:",width)

	# g10 = bef_gray[0:225, first_pix_loc1_x:]
	g10 = g_bk[y1+50:y2, first_pix_loc1_x+30:width-100]

	# stack_img = np.empty_like(g10)

	return g10



def custom_filter(image,k):
	global stack_img
	# global image
	fimg = image.copy()
	# stack_img = []
	# for k in kernel_list:
		# print(k)
	fimg = cv2.filter2D(image,cv2.CV_8UC3,k)

	if debug:
		showimage("filtered",fimg)


	fimg = custom_soften(fimg)

	test_for_lap(fimg)



def continuous_gabor(image):
	"""  sigma: This parameter controls the width of the Gaussian envelope used in the Gabor kernel.4
		lambda:  is the wavelength of the sinusoidal factor in the above equation.
		Gamma:  controls the ellipticity of the gaussian. When gamma = 1, the gaussian envelope is circular.
		psi: This parameter controls the phase offset.

	"""
	# global kernel_list
	global stack_img

	stack_img = np.zeros_like(image)

	ksize = 9
	for theta in np.arange(0,np.pi,np.pi/8):
		kernel = cv2.getGaborKernel((ksize,ksize), 1.0, theta, 5.8, 0.5, 0, ktype = cv2.CV_32F)
		kernel /= 1.5*kernel.sum()
		# kernel_list.append(kernel)
		# print("**************theta**********::",theta)
		custom_filter(image,kernel)

	white_pixels = np.where(stack_img==255)
	white_pixel_value = len(white_pixels[0])
	print("white_pixels: ",white_pixel_value)



	if white_pixel_value>50 and debug == True:
		showimage("final stack",stack_img)

	if 0:
		showimage("final stack",original_image)
		cv2.imshow("final stack",stack_img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	# if debug:
	# 	showimage("final stack",original_image)

	return white_pixel_value



if __name__ == "__main__":

	time_taken = time.time()
	# path = "rpi-7_images/big_shaft_15/"
	# path = "analysis/lap_test/"
	# path = "analysis/ok_small_shaft/"
	# path = "analysis/gripper_test_new/"
	# path = "analysis/failure_test/"
	path = "analysis/dents-cam-3/"

	images=[]

	for r,d,f in os.walk(path):
		for files in f:
			if ".jpg" in files:
				images.append(os.path.join(r,files))

	for img in images:

		image_name= img

		# image_name = "analysis/Webcam157.jpg"
		# image_name = "analysis/Webcam167.jpg"
		# image_name = "analysis/lap_test/Webcam1579.jpg"	
		# image_name = "analysis/lap_test/Webcam1621.jpg"	
		# image_name = "analysis/lap_test/Webcam1659.jpg"	
		# image_name = "analysis/lap_test/Webcam1607.jpg"	

		original_image = cv2.imread(image_name,0)

		print("*******************************"+str(image_name)+"********************************")

		# showimage(img,original_image)

		# clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(5,5))
		# clahe = clahe.apply(original_image)

		# if debug:
		# 	showimage("clahe",clahe)

		# clahe = cv2.equalizeHist(clahe)

		# if debug:
		# 	showimage("hist",clahe)

		crop_image = dynamic_crop(original_image)

		if debug:
			showimage('clahe',crop_image)


		# clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(5,5))
		# clahe = clahe.apply(crop_image)

		# if debug:
		# 	showimage("clahe",clahe)

		continuous_gabor(crop_image)	

		# t2 = time.time()

		time_taken =  time.time()-time_taken


		print("time taken: ",time_taken)	

		break	