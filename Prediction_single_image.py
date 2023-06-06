from metrics import dice_coef, iou, dice_loss
import tensorflow as tf
import cv2
import numpy as np

def patchify(img, patch_size):
    patches = []
    for i in range(0, img.shape[0], patch_size):
        for j in range(0, img.shape[1], patch_size):
            patch = img[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
    return np.array(patches)

def unpatchify(patches, img_size):
    i, j = 0, 0
    unpatched = np.zeros(img_size)
    for patch in patches:
        unpatched[i:i + patch.shape[0], j:j + patch.shape[1]] = patch
        if j + patch.shape[1] < img_size[1]:
            j += patch.shape[1]
        else:
            i += patch.shape[0]
            j = 0
    return unpatched

def predict_and_save_mask(image_path, model_path, output_path, patch_size=256):
    img = cv2.imread(image_path, 1)

    SIZE_X = (img.shape[1]//patch_size)*patch_size 
    SIZE_Y = (img.shape[0]//patch_size)*patch_size 

    print(SIZE_X,SIZE_Y)

    large_img = img[:SIZE_Y, :SIZE_X]

    large_img = large_img/255

    patches = patchify(large_img, patch_size)
    patches = patches.reshape((-1, patch_size, patch_size, 3))

    model = tf.keras.models.load_model(model_path, custom_objects={'dice_coef': dice_coef, 'iou': iou, 'dice_loss': dice_loss})

    pred_mask =  model.predict(patches)
    pred_mask = np.argmax(pred_mask, axis=-1)

    predicted_patches_reshaped = pred_mask.reshape((-1, patch_size, patch_size))
    unpatched_prediction = unpatchify(predicted_patches_reshaped, (SIZE_Y, SIZE_X))

    cv2.imwrite(output_path, unpatched_prediction*255)

# Call the function
predict_and_save_mask(
    image_path='droneimages_73/IMG_210413_155838_0131.tif', 
    model_path="files/model.h5", 
    output_path='droneimages_73/IMG_210413_155838_0131_mask.tif'
)
