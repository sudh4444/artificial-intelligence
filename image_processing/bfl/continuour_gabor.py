import cv2
import numpy as np
from envision.convolution import convolve_sobel
import time


kernel_list = []
debug = False
stack_img = []

def showimage(image_name,image):
	cv2.imshow(image_name,image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def test_for_lap(image):
	global stack_img

	blur = cv2.medianBlur(image,3)

	# if debug:
	# 	showimage("blur",blur)

	sobel = convolve_sobel(img=blur,
                   threshold=50,
                   sobel_kernel_left_right=True,
                   sobel_kernel_right_left=True,
                   sobel_kernel_top_bottom=True,
                   sobel_kernel_bottom_top=True,
                   sobel_kernel_diagonal_top_left=False,
                   sobel_kernel_diagonal_bottom_left=False,
                   sobel_kernel_diagonal_top_right=False,
                   sobel_kernel_diagonal_bottom_right=False)
	sobel = cv2.convertScaleAbs(sobel)

	if debug:
		showimage("sobel",sobel)


	gaus = cv2.GaussianBlur(sobel, (15,15), -1)

	if debug:
		showimage("Gaussian",gaus)

	_,thresh = cv2.threshold(gaus,200,255,cv2.THRESH_BINARY)

	if debug:
		showimage("thresh",thresh)

	stack_img = cv2.bitwise_or(stack_img,thresh)

	if debug:
		showimage("stack image",stack_img)
		
	



def dynamic_crop(g_bk):

	g9 = g_bk.copy()

	g9 = cv2.medianBlur(g9,49)

	if debug:
		showimage("blur",g9)

	_,g9 = cv2.threshold(g9,10,255,cv2.THRESH_BINARY)

	if debug:
		showimage("thresh",g9)

	first_pix_loc1_x = 0
	y1 = 220  + 20
	y2 = 520  - 50-20
	x_fixed = 200
	
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
	g10 = g_bk[y1:y2, first_pix_loc1_x+20:]

	return g10



def custom_filter(image):
	global stack_img
	# global image
	fimg = image.copy()
	stack_img = np.empty_like(fimg)
	for k in kernel_list:
		# print(k)
		fimg = cv2.filter2D(image,cv2.CV_8UC3,k)

		if debug:
			showimage("filtered",fimg)

		test_for_lap(fimg)

	cv2.imshow("final stack",stack_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	if debug:
		showimage("final stack",stack_img)





def continuous_gabor(image):
	"""  sigma: This parameter controls the width of the Gaussian envelope used in the Gabor kernel.4
		lambda:  is the wavelength of the sinusoidal factor in the above equation.
		Gamma:  controls the ellipticity of the gaussian. When gamma = 1, the gaussian envelope is circular.
		psi: This parameter controls the phase offset.

	"""
	global kernel_list

	ksize = 9
	for theta in np.arange(0,np.pi,np.pi/8):
		kernel = cv2.getGaborKernel((ksize,ksize), 1.0, theta, 4.8, 0.5, 0, ktype = cv2.CV_32F)
		kernel /= 1.5*kernel.sum()
		kernel_list.append(kernel)
		# print("**************theta**********::",theta)
	custom_filter(image)



if __name__ == "__main__":

	t1 = time.time()
	# print("time t1",t1)

	# print("pi value: ",np.pi)

	path = "analysis/lap_test/"
	image_name = "Webcam1629.jpg"

	image = cv2.imread(path+image_name,0)

	if debug:
		showimage(image_name,image)

	crop_image = dynamic_crop(image)

	clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(5,5))
	crop_image = clahe.apply(crop_image)

	if debug:
		showimage('clahe',crop_image)



	continuous_gabor(crop_image)

	t2 = time.time()

	time_taken = t2-t1

	print("time taken: ",time_taken)