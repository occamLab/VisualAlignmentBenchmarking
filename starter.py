#!/usr/bin/env python3

#/home/powerhorse/Documents/visualAlignmentComparison/1635895978.43579

# TODO: Add documentation and create a way to handle outputs and merge them into one dataframe.

# Running this file will use several different visual alignment methods on a set of camera/alignment images and generate several comparison photos. The path above is an example path to one of the files you can run this visual alignment comparison on. The output can be found in the output folder in the same parent folder as this file under the long number name at the end of the path mentioned above.
 
from os import system
import sys
from glob import glob
import os
import re

#Splits the metadata path to get its id (2nd to last character)
def get_path_id(path):
    norm_path = os.path.normpath(path)
    path_list = norm_path.split(os.sep)
    path_id = path_list[-2]
    return path_id

#Splits the metadata path to get the image id (last character) with "\d" + last character
def get_image_id(path):
    norm_path = os.path.normpath(path)
    path_list = norm_path.split(os.sep)
    image_name = path_list[-1]
    image_id = re.findall("\d+", image_name)
    return image_id[0]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("USAGE: ./starter.py pathToDataDir")
        sys.exit(1)

    for metadata_path in glob(sys.argv[1] + '/**/camera*metadata.json', recursive=True):
        base_dir = "output"
        path_id = get_path_id(metadata_path)
        image_id = get_image_id(metadata_path)

        # construct paths to:
        camera_metadata = metadata_path
        align_metadata = camera_metadata.replace('cameraimage', 'alignimage')
        camera_image = camera_metadata.replace('_metadata.json', '.jpg')
        align_image = align_metadata.replace('_metadata.json', '.jpg')

        #TODO: some images are not present. Maybe saving them in local storage and uploading later. 
        
        total_exp = 5 # TODO: make this a dict that has this

        for vf_id in range(total_exp):
            output_path = os.path.join(base_dir, path_id, image_id, str(vf_id))
            
            # Check whether the specified path exists or not
            isExist = os.path.exists(output_path)

            if True: #not isExist:
                # Create a new directory because it does not exist 
                if not isExist:
                    os.makedirs(output_path)
                print(output_path)

                # only run visual alignment if the folder didn't exist before because we don't want to waste energy rerunning the alg
                system('./build/visual_alignment --align-image-path ' + align_image + ' --camera-image-path ' + 
                        camera_image + ' --camera-metadata-path ' + camera_metadata + ' --align-metadata-path ' + 
                        align_metadata + ' --vftype ' + str(vf_id) + ' --output-directory ' + output_path + ' --debug false')

                
            # invoke superpoint as a separate process, copy match pairs and put it in the same folder

            # TODO: merge into existing spreadsheet or just make a program that goes through all the results and compiles it into a dataframe.
