import cv2
import random
import os,sys



sr = cv2.dnn_superres.DnnSuperResImpl_create()

def blur(img):
	blurImg = cv2.blur(img,(3,3))
	return blurImg

def espcn(img):
	
	path = "ESPCN_x2.pb"
	sr.readModel(path)
	sr.setModel("espcn", 2) # set the model by passing the value and the upsampling ratio
	result = sr.upsample(img) # upscale the input image
	return result
	


def pyrup(img):
	
	rows, cols, _channels = map(int, img.shape)

	img = cv2.pyrUp(img, dstsize=(2 * cols, 2 * rows))

	return img

j=0
for i in os.listdir("./input"):
	i = "input/"+i
	print(i)
	img = cv2.imread(i)
	flag = random.randint(1,20)

	if flag%2==0:
		print("blur")
		img = blur(img)
	else:
		print("No blur")

	imge = espcn(img)
	imgp = pyrup(img)

	final_patha = os.path.join("output","out"+str(j)+"a.jpg")
	final_pathe = os.path.join("output","out"+str(j)+"e.jpg")
	final_pathp = os.path.join("output","out"+str(j)+"p.jpg")

	cv2.imwrite(final_patha,img)
	cv2.imwrite(final_pathe,imge)
	cv2.imwrite(final_pathp,imgp)
	j+=1
	

