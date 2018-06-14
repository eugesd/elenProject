# %load cvpr.py

"""
Mask R-CNN

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

import pandas as pd

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Dataset directories
DATASET = os.path.join("/home/eugeniorivera/.kaggle/competitions/cvpr-2018-autonomous-driving/")
       
DATASET_IMAGES = os.path.join(DATASET, "train_color")
DATASET_LABELS = os.path.join(DATASET, "train_label")
DATASET_TEST = os.path.join(DATASET, "test")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CVPRConfig(Config):
    """Configuration for training.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "CVPR"
    GPU_COUNT = 1
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # Background + CVPR WAD Classes
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CVPRDataset(utils.Dataset):

    def load_cvpr(self, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
#         self.add_class("cvpr", 33, "car")
#         self.add_class("cvpr", 34, "motorcycle")
#         self.add_class("cvpr", 35, "bicycle")
#         self.add_class("cvpr", 36, "person")
#         self.add_class("cvpr", 38, "truck")
#         self.add_class("cvpr", 39, "bus")
#         self.add_class("cvpr", 40, "tricycle")
        
        self.add_class("cvpr", 1, "car")
        self.add_class("cvpr", 2, "motorcycle")
        self.add_class("cvpr", 3, "bicycle")
        self.add_class("cvpr", 4, "person")
        self.add_class("cvpr", 5, "truck")
        self.add_class("cvpr", 6, "bus")
        self.add_class("cvpr", 7, "tricycle")
        # Import Dataset


        image_ids = os.listdir(DATASET_IMAGES)
        image_ids.sort()
        label_ids = os.listdir(DATASET_LABELS)
        label_ids.sort()

        df_id = pd.DataFrame()
        df_id['label'] = label_ids
        df_id['image'] = image_ids
        df_id['label_path'] = df_id['label'].apply(lambda x: os.path.join(DATASET_LABELS, x))
        df_id['image_path'] = df_id['image'].apply(lambda x: os.path.join(DATASET_IMAGES, x))


        # There are 39222 images in our train_color directory
        # So lets take 60% of these images for training and the rest for validation
        df_id_train = df_id.iloc[:23533]
        df_id_valid = df_id.iloc[25334:]

        # Train or validation dataset?
        if subset == "train":
            df = df_id_train
        else:
            df = df_id_valid
        #print("DEBUG: Adding images")
        for index, row in df.iterrows():
           #print(row['label'])
           self.add_image(
               "cvpr",
               image_id=str(row['image']),
               path=str(row['image_path']))
        #print(df.iloc[:4])


    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
           one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        print("DEBUG: LOADING MASK")

        image_info = self.image_info[image_id]
        image_fn = image_info["id"]
        #print("DEBUG: image_fn: " + str(image_fn))
        #TODO: Check to see if this works
        label_id = os.path.splitext(image_fn)[0] + "_instanceIds.png"
        label_path = os.path.join(DATASET_LABELS, label_id)
        label = skimage.io.imread(label_path)
        class_map = np.array([0,33,34,35,36,38,39,40])
        for i,row in enumerate(label//1000):
            for j,p in enumerate(row):
                try:
                    assert p in class_map
                except:
                    label[i,j] = 0
                    #print("Not in class_map")

        class_ids = np.unique(label)
        mask = np.zeros((label.shape[0],label.shape[1],len(class_ids)))

        i=0
        for instance in class_ids:
            mask[:,:,i] =  (label == instance).astype(int)
            i = i + 1
            #print('.', end='')

        class_ids = class_ids//1000
        class_map = np.array([0,33,34,35,36,38,39,40])
        for idx, c in enumerate(class_ids):
            try:
                assert c in class_map
                if c == 33:
                    class_ids[idx] = 1
                elif c == 34:
                    class_ids[idx] = 2
                elif c == 35:
                    class_ids[idx] = 3
                elif c == 36:
                    class_ids[idx] = 4
                elif c == 38:
                    class_ids[idx] = 5
                elif c == 39:
                    class_ids[idx] = 6
                elif c == 40:
                    class_ids[idx] = 7
                else:
                    print('error')
                    class_ids[idx] = 0
            except:
                class_ids[idx] = 0
                print('error')
        print(class_ids)
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        #TODO: ?
        info = self.image_info[image_id]
        if info["source"] == "cvpr":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CVPRDataset()
    dataset_train.load_cvpr("train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CVPRDataset()
    dataset_val.load_cvpr("val")
    dataset_val.prepare()

    # Testing 
    #     image_entry = dataset_train.image_info[0]
    #     print(type(image_entry['id']))
    #     dataset_train.load_mask(image_entry['id'])

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("NOT Training network heads")
    model.train(dataset_train, dataset_val,
               learning_rate=config.LEARNING_RATE,
               epochs=30,
               layers='heads')


def apply_mask(image, mask):
    # Color gray
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        mask_applied = np.where(mask, image, gray).astype(np.uint8)
    else:
        mask_applied = gray.astype(np.uint8)
    return mask_applied

def detect_cvpr(model, image_path=None):
    # Read in image
    PRESET_IMAGE = os.path.join(DATASET_TEST,"4087277ed190606ca20478cfd302b52d.jpg")
    if image_path == None:
        print("Detecting preset image located at:",PRESET_IMAGE)
        image = skimage.io.imread(PRESET_IMAGE)
    else:
        print("Detecting image located at:",image_path)
        image = skimage.io.imread(image_path)
    
    # Run detection
    r = model.detect([image], verbose=1)[0]
    
    # Apply mask to image
    mask = apply_mask(image, r['masks'])
    
    # Save Mask
    file_name = "mask_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, mask)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse
    DEFAULT_WEIGHTS_DIR = '../../'
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect CVPR objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'run'")
    parser.add_argument('--weights', required=False,
                        default=DEFAULT_WEIGHTS_DIR,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--image', required=False,
                        metavar="path to image",
                        help="Image to generate mask for")
    args = parser.parse_args()
    
    print("Command: ", args.command)
    print("Weights: ", args.weights)

    # Setting up Configuration
    if args.command == "train":
        config = CVPRConfig()
    else:
        class InferenceConfig(CVPRConfig):
            # Batch size to 1 since running inferece on one img at time
            # Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()
    
    # Creating model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

    print('coco: ',COCO_WEIGHTS_PATH)
    
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        weights_path = model.find_last()[1]
    else:
        weights_path = args.weights
    
    # Loading weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude last layers since require matching num of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    if args.command == "train":
        train(model)
        pass
    elif args.command == "run":
        detect_cvpr(model)
    else:
        print('try "python3 cvpr.py train" or "python3 cvpr.py run"')
    #train(model)
# Default below:

#     # Train or evaluate
#     if args.command == "train":
#         train(model)
#     elif args.command == "splash":
#         detect_and_color_splash(model, image_path=args.image,
#                                 video_path=args.video)
#     else:
#         print("'{}' is not recognized. "
#               "Use 'train' or 'splash'".format(args.command))
#sys.exit(0)
