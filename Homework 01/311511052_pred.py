# %%
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

# %%
# load model
model = load_model('resnet50.h5')

# %%
# load image from image_path_list.txt
with open('image_path_list.txt', 'r') as f:
    image_path_list = f.readlines()

# %%
# clear 311511052.txt
with open('311511052.txt', 'w') as f:
    f.write('')

# %%
# save result in 311511052.txt
with open('311511052.txt', 'a') as f:
    for image_path in image_path_list:
        image_path = image_path.strip()
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # predict the image and write the result in 311511052.txt
        preds = model.predict(x)
        # write without change line or space just wtite concate
        # write only the number of the class
        if preds > 0.5:
            f.write('1')
        else:
            f.write('0')


