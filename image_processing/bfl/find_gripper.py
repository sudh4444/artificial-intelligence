import os 
import cv2
import numpy as np

image_name = ""
count_index= 0
first_pix_loc1_x=0

def showimage(image_name,image):
	# return
	cv2.imshow(image_name,image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def dynamic_crop(g9,g_bk):
	
	global first_pix_loc1_x

	if 1:#debug:
			#showimage("ImageWindow",g9)
			cv2.imwrite("outputs/g9_"+str(count_index)+".jpg",g9)

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


	if 1:
		showimage('ImageWindow', g9)
		# cv2.waitKey(0)





	# g10 = bef_gray[0:225, first_pix_loc1_x:]
	g10 = g_bk[y1+30:y2-20, first_pix_loc1_x+20:]

	return g10



def identify_gripper(image):
	global image_name
	global count_index
	global first_pix_loc1_x

	print("Applying filter B for Gripper")
	debug = True

	try:
		crop1_x1 = 100 + 0
		crop1_x2 = 1200 - 0
		crop1_y1 = 220  + 0 - 20 -50
		crop1_y2 = 520  + 50 + 20

		mid_y = int(round((crop1_x1 - crop1_x2) / 2))
		counter_y = 0

		# image = cv2.imread(str(path) + str(image_name))
		col_img = image.copy()
		# image=original_image.copy()
		gray = image[crop1_y1:crop1_y2, crop1_x1:crop1_x2]

		if debug:
			showimage("crop",gray)


		bef_gray = gray.copy()




		# #-----Converting image to LAB Color model----------------------------------- 
		# lab= cv2.cvtColor(gray, cv2.COLOR_BGR2LAB)
		# # cv2.imshow("lab",lab)

		# #-----Splitting the LAB image to different channels-------------------------
		# l, a, b = cv2.split(lab)
		# # cv2.imshow('l_channel', l)
		# # cv2.imshow('a_channel', a)
		# # cv2.imshow('b_channel', b)

		# #-----Applying CLAHE to L-channel-------------------------------------------
		# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
		# cl = clahe.apply(l)
		# # cv2.imshow('CLAHE output', cl)

		# #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
		# limg = cv2.merge((cl,a,b))
		# # cv2.imshow('limg', limg)

		# #-----Converting image from LAB Color model to RGB model--------------------
		# final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
		# # cv2.imshow('final', final)


		# # gray = final






		# if debug:
		# 	cv2.imwrite("outputs/"+str(image_name)+"_A1_"+"crop.jpg", image)
		# 	showimage('ImageWindow',image)
		# 	cv2.waitKey(0)





		gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY) # grayscale

		clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(5,5))
		gray = clahe.apply(gray)

		# cv2.imwrite("image_clahe_output.jpg",cl1)

		if debug:
			showimage("clahe",gray)

		g_bk = gray.copy()


		# if debug:
		# 	cv2.imwrite("outputs/"+str(image_name)+"_A2_"+"gray.jpg", gray)
		# 	showimage('gray image', image)
		# 	cv2.waitKey(0)



		gray = cv2.medianBlur(gray, 89)


		if debug:
			cv2.imwrite("outputs/"+str(image_name)+"_A3_"+"blur.jpg", gray)
			showimage('median blur', gray)
			cv2.waitKey(0)




		_,g9 = cv2.threshold(gray,25,255,cv2.THRESH_BINARY)

		# showimage("thresh",g9)
		
		global count_index
		count_index+=1

		g10 = dynamic_crop(g9,g_bk)

		

		if debug:
			showimage('dynamic crop', g10)
			cv2.waitKey(0)



		# g12 = cv2.equalizeHist(g10)

		# if debug:
		# 	showimage("histogram",g12)



		# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		# g12 = clahe.apply(g10)

		# # cv2.imwrite("image_clahe_output.jpg",cl1)

		# if debug:
		# 	showimage("clahe",g12)

		g12 = cv2.medianBlur(g10,7)

		if debug:
			showimage("medianBlur",g12)


		g_kernel2 = cv2.getGaborKernel((7,7), 1.55, np.pi, 5.45, 0.2, 0, ktype=cv2.CV_32F) #(ksize, sigma, theta, lambda, gamma, psi, ktype)
		g13 = cv2.filter2D(g12, cv2.CV_8UC3, g_kernel2)


		if debug:
			showimage('gabor filter', g13)
			cv2.waitKey(0)


		g13 = cv2.GaussianBlur(g13,(7,7),0)

		# showimage("GaussianBlur",g13)

		# pos = np.where(g13>80)

		# g13[pos]=255

		# showimage("clahe mod",g13)

		_,g14 = cv2.threshold(g13,160,255,cv2.THRESH_BINARY)

		if debug:
			showimage("gabor thresh",g14)

		

		




		# g10 = g10[110:170, 800:750+150]

		# if debug:
		# 	showimage('ImageWindow', g10)
		# 	cv2.waitKey(0)



		# g12 =  cv2.medianBlur(g12, 7)


		# if debug:
		# 	showimage('medianBlur', g12)
		# 	cv2.waitKey(0)






		# edges = cv2.Canny(g12,100,255)

		# if debug:
		# 	showimage('ImageWindow', edges)
		# 	cv2.waitKey(0)




		# g_kernel2 = cv2.getGaborKernel((7, 7), 1.7, np.pi/3.4, 4.1, 0.5, 0, ktype=cv2.CV_32F)
		


		# g13 = cv2.GaussianBlur(g13,(5,5),0)


		


		# # g14 =  cv2.medianBlur(g14, 7)


		n_black_pix = np.sum(g14 <100)
		print('\t\tNumber of black pixels for gripper:', n_black_pix)
		print('\t\tfirst_pix_loc1_x:', first_pix_loc1_x)


		if n_black_pix>200:
			print("gripper present",n_black_pix)


		# # if n_black_pix>5:# or first_pix_loc1_x<10:
		# if n_black_pix>5 and first_pix_loc1_x>10 and first_pix_loc1_x<200:

		# 	cv2.rectangle(img = col_img, pt1 = (920-20, 260-20), pt2 = (1100+20, 450+20), color = (0, 0, 255), thickness = 2)

		# 	font                   = cv2.FONT_HERSHEY_SIMPLEX
		# 	bottomLeftCornerOfText = (920-20-300,260-20+50)
		# 	fontScale              = 1
		# 	fontColor              = (0,0,255)
		# 	lineType               = 2

		# 	cv2.putText(col_img,'NOT OK - Gripper', 
		# 	    bottomLeftCornerOfText, 
		# 	    font, 
		# 	    fontScale,
		# 	    fontColor,
		# 	    lineType)


		# 	# showimage('ImageWindow', g14)
		# 	showimage('ImageWindow2', col_img)
		# 	# showimage('ImageWindow3', g9)


		# if debug:
		# 	showimage('ImageWindow', g14)
		# 	cv2.waitKey(0)


		return ((first_pix_loc1_x, n_black_pix, col_img))



		# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
		# g15 = cv2.morphologyEx(g14, cv2.MORPH_OPEN, kernel)


		# if debug:
		# 	showimage('ImageWindow', g15)
		# 	cv2.waitKey(0)

		return

	except Exception as e:
		print ("Exception", e)

	return


if __name__ == "__main__":
	# path = "rpi-7_images/big_shaft_15/"
	path = "analysis/gripper_test_new/"

	images=[]

	for r,d,f in os.walk(path):
		for files in f:
			if ".jpg" in files:
				images.append(os.path.join(r,files))

	for img in images:

		image_name= img

		# image_name = "analysis/Webcam157.jpg"
		# image_name = "analysis/Webcam167.jpg"
		# image_name = "analysis/Webcam1213.jpg"

		original_image = cv2.imread(image_name)

		print("*******************************"+str(image_name)+"********************************")

		# showimage(img,original_image)

		identify_gripper(original_image)

		# break	