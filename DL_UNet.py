import arcpy
from arcpy.ia import *
import numpy as np
import os
import json
import shutil
import rasterio

# workspace
arcpy.env.workspace = r"C:\Users\workspace"

#Define the file path 
input_raster = r"D:\roadpath\raster.tif"
training_samples = r"D:\roadpath\samples.shp"
output_training_data = r"D:\roadpath\output_data_file"
labels_folder = r"D:\roadpath\output_data_file\labels"
output_model = r"D:\roadpath\outputmodel"
model_emd = r"D:\roadpath\outputmodel\your_model.emd"
output_classified_raster = r"D:\roadpath\.gdb\raster"

if os.path.exists(output_model) and os.listdir(output_model):
    shutil.rmtree(output_model)
    os.makedirs(output_model)
elif not os.path.exists(output_model):
    os.makedirs(output_model)

# spatial reference
reference_system = arcpy.SpatialReference(6708)#It need to be modified.

#1 Export Training Data For Deep Learning
arcpy.ia.ExportTrainingDataForDeepLearning(
    in_raster=input_raster,
    out_folder=output_training_data,
    in_class_data=training_samples,
    image_chip_format="TIFF",
    tile_size_x=256,
    tile_size_y=256,
    stride_x=128,
    stride_y=128,
    output_nofeature_tiles="ONLY_TILES_WITH_FEATURES",
    #buffer_radius=0,
    metadata_format="Classified_Tiles",
    class_value_field="Classvalue", # classify fields in samples data
    rotation_angle=45,
    # in_mask_polygons=input_mask,
    reference_system="MAP_SPACE",
    #processing_mode="PROCESS_AS_MOSAICKED_IMAGE",
    blacken_around_feature="NO_BLACKEN",
    crop_mode="FIXED_SIZE",
    #in_raster2=none,
    min_polygon_overlap_ratio=0.2,
)
print("Training data exported to:", output_training_data)

#2 Training the deep learning model
# 2.1 computing class weight
def compute_class_weights(labels_folder, class_ids):
    all_pixels = []
    for filename in os.listdir(labels_folder):
        if filename.endswith(".tif"):
            path = os.path.join(labels_folder, filename)
            with rasterio.open(path) as src:
                label = src.read(1)
                all_pixels.append(label.flatten())

    all_pixels = np.concatenate(all_pixels)
    # Count each class
    counts = {cid: np.sum(all_pixels == cid) for cid in class_ids}
    print("\nClass Pixel Counts:")
    for cid in class_ids:
        print(f"Class {cid}: {counts[cid]} pixels")

    # Assign class weights (Median Frequency Balancing)
    count_array = np.array([counts[cid] for cid in class_ids])
    median_count = np.median(count_array)
    weights = {cid: float(median_count / counts[cid]) if counts[cid] > 0 else 0.0
               for cid in class_ids}
    print("\nAssigned Class Weights:")
    for cid in class_ids:
        print(f"  Class {cid}: weight = {weights[cid]:.4f}")

    # Convert to JSON
    class_weights_json = json.dumps(weights)

    return weights, class_weights_json

# 2.2 setting parameters
augmentation_parameters = {
    "do_flip": True,
    "flip_vert": True,
    "max_rotate": 10,
    "max_zoom": 1.1,
    "max_lighting": 0.3,
    "max_warp": 0.02,
    "p_affine": 0.5,
    "p_lighting": 0.5
}

arguments_p = [
    ["class_balancing", "False"],
    ["class_weights", class_weights_json],
    ["dice_loss_average", "micro"],
    ["dice_loss_fraction", 0],
    ["focal_loss", "True"],
    ["mixup", "True"]
]

augmentation_p = json.dumps(augmentation_parameters)
print(arguments_p)
print(augmentation_parameters)
print(type(arguments_p))

#2.3 training model
arcpy.ia.TrainDeepLearningModel(
    in_folder=output_training_data,
    out_folder=output_model,
    model_type="UNET",
    max_epochs=50,
    batch_size=64,
    # arguments=arguments_parameters,
    #pretrained_model="",
    learning_rate=0.0001,
    backbone_model="RESNET50",
    validation_percentage=20,
    stop_training= "STOP_TRAINING",
    freeze = "UNFREEZE_MODEL",
    augmentation = "CUSTOM",
    augmentation_parameters =augmentation_p,
    chip_size = 256,
    # resize_to: str = "#",
    weight_init_scheme = "ALL_RANDOM",
    monitor = "VALID_LOSS"
)
print("Model trained and saved to:", output_model)

#3 classify pixels using deep learning
classified_raster = arcpy.ia.ClassifyPixelsUsingDeepLearning(
    in_raster=input_raster,
    out_classified_folder = output_classified_raster,
    in_model_definition=model_emd,
    out_featureclass=output_feature,
    arguments={"padding": 64,
               "batchsize":4,
               "predict_background": True},
    processing_mode="PROCESS_AS_MOSAICKED_IMAGE",
)
print("Classification completed. Output saved to:", output_classified_raster)

