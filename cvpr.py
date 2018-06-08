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
   """Configuration for training on the toy  dataset.
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
   STEPS_PER_EPOCH = 100

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
       self.add_class("cvpr", 33, "car")
       self.add_class("cvpr", 34, "motorcycle")
       self.add_class("cvpr", 35, "bicycle")
       self.add_class("cvpr", 36, "person")
       self.add_class("cvpr", 38, "truck")
       self.add_class("cvpr", 39, "bus")
       self.add_class("cvpr", 40, "tricycle")
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
       print("DEBUG: Adding images")
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
           except:
               class_ids[idx] = 0
               print('error')
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


def color_splash(image, mask):
   """Apply color splash effect.
   image: RGB image [height, width, 3]
   mask: instance segmentation mask [height, width, instance count]
   Returns result image.
   """
   # Make a grayscale copy of the image. The grayscale copy still
   # has 3 RGB channels, though.
   gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
   # We're treating all instances as one, so collapse the mask into one layer
   mask = (np.sum(mask, -1, keepdims=True) >= 1)
   # Copy color pixels from the original color image where mask is set
   if mask.shape[0] > 0:
       splash = np.where(mask, image, gray).astype(np.uint8)
   else:
       splash = gray
   return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
   assert image_path or video_path

   # Image or video?
   if image_path:
       # Run model detection and generate the color splash effect
       print("Running on {}".format(args.image))
       # Read image
       image = skimage.io.imread(args.image)
       # Detect objects
       r = model.detect([image], verbose=1)[0]
       # Color splash
       splash = color_splash(image, r['masks'])
       # Save output
       file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
       skimage.io.imsave(file_name, splash)
   elif video_path:
       import cv2
       # Video capture
       vcapture = cv2.VideoCapture(video_path)
       width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
       height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
       fps = vcapture.get(cv2.CAP_PROP_FPS)

       # Define codec and create video writer
       file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
       vwriter = cv2.VideoWriter(file_name,
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 fps, (width, height))

       count = 0
       success = True
       while success:
           print("frame: ", count)
           # Read next image
           success, image = vcapture.read()
           if success:
               # OpenCV returns images as BGR, convert to RGB
               image = image[..., ::-1]
               # Detect objects
               r = model.detect([image], verbose=0)[0]
               # Color splash
               splash = color_splash(image, r['masks'])
               # RGB -> BGR to save image to video
               splash = splash[..., ::-1]
               # Add image to video writer
               vwriter.write(splash)
               count += 1
       vwriter.release()
   print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
   import argparse
   print("Checkpoint1")
   # Setup Configuration
   config = CVPRConfig()
   config.display()
   print("Checkpoint2")
   # Load Training Model
   model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)
   print("Checkpoint3")
   # Load COCO weights
   weights_path = COCO_WEIGHTS_PATH
   if not os.path.exists(weights_path):
       utils.download_trained_weights(weights_path)
   
   model.load_weights(weights_path, by_name=True, exclude=[
       "mrcnn_class_logits", "mrcnn_bbox_fc",
       "mrcnn_bbox", "mrcnn_mask"])
   train(model)
#     # Parse command line arguments
#     parser = argparse.ArgumentParser(
#         description='Train Mask R-CNN to detect balloons.')
#     parser.add_argument("command",
#                         metavar="<command>",
#                         help="'train' or 'splash'")
#     parser.add_argument('--dataset', required=False,
#                         metavar="/path/to/balloon/dataset/",
#                         help='Directory of the Balloon dataset')
#     parser.add_argument('--weights', required=True,
#                         metavar="/path/to/weights.h5",
#                         help="Path to weights .h5 file or 'coco'")
#     parser.add_argument('--logs', required=False,
#                         default=DEFAULT_LOGS_DIR,
#                         metavar="/path/to/logs/",
#                         help='Logs and checkpoints directory (default=logs/)')
#     parser.add_argument('--image', required=False,
#                         metavar="path or URL to image",
#                         help='Image to apply the color splash effect on')
#     parser.add_argument('--video', required=False,
#                         metavar="path or URL to video",
#                         help='Video to apply the color splash effect on')
#     args = parser.parse_args()

#     # Validate arguments
#     if args.command == "train":
#         assert args.dataset, "Argument --dataset is required for training"
#     elif args.command == "splash":
#         assert args.image or args.video,\
#                "Provide --image or --video to apply color splash"

#     print("Weights: ", args.weights)
#     print("Dataset: ", args.dataset)
#     print("Logs: ", args.logs)

#     # Configurations
#     if args.command == "train":
#         config = BalloonConfig()
#     else:
#         class InferenceConfig(BalloonConfig):
#             # Set batch size to 1 since we'll be running inference on
#             # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
#             GPU_COUNT = 1
#             IMAGES_PER_GPU = 1
#         config = InferenceConfig()
#     config.display()

#     # Create model
#     if args.command == "train":
#         model = modellib.MaskRCNN(mode="training", config=config,
#                                   model_dir=args.logs)
#     else:
#         model = modellib.MaskRCNN(mode="inference", config=config,
#                                   model_dir=args.logs)

#     # Select weights file to load
#     if args.weights.lower() == "coco":
#         weights_path = COCO_WEIGHTS_PATH
#         # Download weights file
#         if not os.path.exists(weights_path):
#             utils.download_trained_weights(weights_path)
#     elif args.weights.lower() == "last":
#         # Find last trained weights
#         weights_path = model.find_last()[1]
#     elif args.weights.lower() == "imagenet":
#         # Start from ImageNet trained weights
#         weights_path = model.get_imagenet_weights()
#     else:
#         weights_path = args.weights

#     # Load weights
#     print("Loading weights ", weights_path)
#     if args.weights.lower() == "coco":
#         # Exclude the last layers because they require a matching
#         # number of classes
#         model.load_weights(weights_path, by_name=True, exclude=[
#             "mrcnn_class_logits", "mrcnn_bbox_fc",
#             "mrcnn_bbox", "mrcnn_mask"])
#     else:
#         model.load_weights(weights_path, by_name=True)

#     # Train or evaluate
#     if args.command == "train":
#         train(model)
#     elif args.command == "splash":
#         detect_and_color_splash(model, image_path=args.image,
#                                 video_path=args.video)
#     else:
#         print("'{}' is not recognized. "
#               "Use 'train' or 'splash'".format(args.command))
