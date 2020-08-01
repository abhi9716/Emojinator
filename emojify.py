import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D,MaxPool2D
from keras import backend as K
from PIL import Image
from skimage import transform
from preprocess_img import create_mask, get_bounding_rect
import cv2
import dlib, os
from imutils import face_utils
from imutils.face_utils import FaceAligner

SHAPE_PREDICTOR_68 = "shape_predictor_68_face_landmarks.dat"
shape_predictor_68 = dlib.shape_predictor(SHAPE_PREDICTOR_68)

detector = dlib.get_frontal_face_detector()
img_width, img_height = 100, 100

def get_emojis():
    emojis_folder = 'emojis/'
    emojis = []
    for emoji in range(len(os.listdir(emojis_folder))):
        emojis.append(cv2.imread(emojis_folder+str(emoji)+'.png', -1))
    return emojis

def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))



def blend(image, emoji, position):
    x, y, w, h = position      
    emoji = cv2.resize(emoji, (w, h))
    try:
        image[y:y+h, x:x+w] = blend_transparent(image[y:y+h, x:x+w], emoji)
    except:
        pass
    return image


def create_model():
	if K.image_data_format() == 'channels_first':
		input_shape = (1, img_width, img_height)
	else:
		input_shape = (img_width, img_height, 1)

	model = Sequential()

	model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
		     activation ='relu', input_shape = input_shape))
	model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
		     activation ='relu'))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Dropout(0.25))


	model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
		     activation ='relu'))
	model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
		     activation ='relu'))
	model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.25))


	model.add(Flatten())
	model.add(Dense(256, activation = "relu"))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation = "softmax"))
	return model

model = create_model()
model.load_weights('model.h5')

def load(np_image):
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (100, 100, 1))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

cam = cv2.VideoCapture(1)
if cam.read()[0]==False:
    cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
fa = FaceAligner(shape_predictor_68, desiredFaceWidth=250)


def emojify(model):
	emojis = get_emojis()
	disp_probab, disp_class = 0, 0
	while True:
		img = cam.read()[1]
		img = cv2.flip(img, 1)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = detector(gray)
		if len(faces) > 0:
			for i, face in enumerate(faces):
				shape_68 = shape_predictor_68(img, face)
				shape = face_utils.shape_to_np(shape_68)
				mask = create_mask(shape, img)
				masked = cv2.bitwise_and(gray, mask)
				maskAligned = fa.align(mask, gray, face)
				faceAligned = fa.align(masked, gray, face)
				(x0, y0, x1, y1) = get_bounding_rect(maskAligned)
				faceAligned = faceAligned[y0:y1, x0:x1]
				cv2.imshow('faceAligned', faceAligned)
				(x, y, w, h) = face_utils.rect_to_bb(face)
				cv2.imshow('face #{}'.format(i), img[y:y+h, x:x+w])
				np_image = load(faceAligned)
				pred_probab = model.predict(np_image)
				pred_class = np.argmax(pred_probab,axis = 1)
				img = blend(img, emojis[int(pred_class)], (x, y, w, h))
		cv2.imshow('img', img)
		if cv2.waitKey(1) == ord('q'):
			break

emojify(model)


