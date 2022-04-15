This readme contains the information on what each python file does including inputs and outputs.

starter.py:

To run the command type: python ./starter.py /home/powerhorse/Documents/visualAlignmentComparison/1635895978.43579
- The long number is the folder that holds the picture data of a specific route. You can find which specific route it is in Google Drive folder.

The code will take every picture in the folder and run the different Visual Alignment algorithms. It then stores the data in a folder (/VisualAlignmentComparison/output/1635895978.43579) where each image has a folder with five subfolders for each algorithm. These subfolders have the image as well as the .json file. 

process_matches.py:

This code is used for manual comparison between the navigation and route image where you can create epipolar lines by clicking on the image. 

process_sfm.py:

Shows points of interest by reprojecting the points onto the image. Currently it's taking the .json from a folder called output_sfm and lists all the points of interest between the images. 

process_sfm_2.py:

Outputs points in matrices? Unsure, but seems to use a similar process to process_sfm.py.


