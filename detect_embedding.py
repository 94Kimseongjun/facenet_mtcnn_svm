import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from keras import backend
from PIL import Image, ImageDraw, ImageFont
from numpy import asarray
from numpy import expand_dims
from numpy import array
from matplotlib import pyplot
import cv2


class FaceAnalysis:
    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        set_session(tf.Session(config=config))
        self.detector = MTCNN()
        self.pre_train_model = load_model('/test/facenet_keras.h5')
        global graph
        graph = tf.get_default_graph()
        print("load OK")

    def extract_face_get_embedding(self, pixels, required_size=(160, 160)):

        # create the detector, using default weights
        try:
            with graph.as_default():

                # detect face in the image
                crop_face = []
                yhat_list = []
                bound_box_list = []
                results = self.detector.detect_faces(pixels)
                if not results:
                    # print("face detect fail")
                    return False, False, False, False

                for result in results:
                    temp_list = []
                    # extract the bounding box from the first face
                    x1, y1, width, height = result['box']
                    # bug fix
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1 + width, y1 + height

                    temp_list.append(x1)
                    temp_list.append(y1)
                    temp_list.append(x2)
                    temp_list.append(y2)

                    bound_box_list.append(temp_list)

                    # extract the face
                    face = pixels[y1:y2, x1:x2]

                    # resize pixels to the model size
                    image = Image.fromarray(face)
                    image = image.resize((160, 160))
                    face_array = asarray(image)
                    crop_face.append(face_array)
                    face_pixels = face_array

                    # scale pixel values
                    face_pixels = face_pixels.astype('float32')

                    # standardize pixel values across channels (global)
                    mean, std = face_pixels.mean(), face_pixels.std()
                    face_pixels = (face_pixels - mean) / std

                    # transform face into one sample
                    samples = expand_dims(face_pixels, axis=0)

                    # make prediction to get embedding
                    yhat = self.pre_train_model.predict(samples)

                    # face_array_list.append(face_array)
                    yhat_list.append(yhat[0])

            return yhat_list, bound_box_list, crop_face, True
        except Exception as e:
            print(e)
            return False, False, False, False
