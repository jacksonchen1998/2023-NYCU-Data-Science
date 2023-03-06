import numpy as np
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':

    if sys.argv[1]:
        # load model
        model = load_model('311511052.h5')

        # load image from image_path_list.txt
        with open(sys.argv[1], 'r') as f:
            image_path_list = f.readlines()

        # clear 311511052.txt
        with open('311511052.txt', 'w') as f:
            f.write('')

        image_path_list

        datagen = ImageDataGenerator(rescale=1./255)

        # open the 311511052.txt file in append mode
        with open('311511052.txt', 'a') as f:
            for index in range(len(image_path_list)):
                img = image.load_img(image_path_list[index].strip(), target_size=(224, 224))
                img_arr = np.expand_dims(image.img_to_array(img), axis=0)
                img_arr = datagen.standardize(img_arr)
                pred = model.predict(img_arr)
                if pred > 0.5:
                    f.write('1')
                else:
                    f.write('0')
            f.close()
    elif not sys.argv[1]:
        print('Please input the text file name.')
    # close the 311511052.txt file