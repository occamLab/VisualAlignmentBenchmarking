This readme contains the information on what each python file does including inputs and outputs.

starter.py:

To run the command type: python ./starter.py /home/powerhorse/Documents/visualAlignmentComparison/1635895978.43579
The long number is the folder that holds the picture data of a specific route. You can find which specific route it is in Google Drive folder.

The code will take every picture in the folder and run the different Visual Alignment algorithms. It then stores the data in a folder (/VisualAlignmentComparison/output/1635895978.43579) where each image has a folder with five subfolders for each algorithm. These subfolders have the image as well as the .json file. It will also tell you real time if the algorithm was able to achieve enough matches to align or if the algorithm was unable to align the images.

Folder # = Algorithm Used
- 0 = SIFT
- 1 = ORB
- 2 = AKAZE MLDB Upright
- 3 = AKAZE MLDB
- 4 = SIFT Upright

process\_matches.py:

This code is used for manual comparison between the navigation and route image where you can create epipolar lines by clicking on the image. Having some issues also running this code--syntax errors with one line in particular. 

process\_sfm.py:

Shows points of interest by reprojecting the points onto the image. Currently it's taking the .json from a folder called output_sfm and lists all the points of interest between the images. 


process\_sfm\_2.py:

Outputs points in matrices? Unsure, but seems to use a similar process to process\_sfm.py. Important to note to that the values within the matrices are not the same values that process\_sfm.py puts out

process\_matches\_opencv.py:

Takes two images and draws the epipolar lines using the SIFT method to find keypoints/descriptors and FLANN parameters. Currently there is already images selected in the code--changing those lines will change what images can be used for this code. The syntax is wrong on line 30. 

process\_matches\_opencv\_backup.py:

Seems to be almost the same exact code as process\_matches\_opencv.py. Need to install the python-tk package?

generate\_csv.py:

It seems to take a .json file and convert it to a csv file. There seems to be a mismatch in the rows to the columns. My guess from looking at the code and at the way the.json result files are structured is that the list(res.values()) in line 50 is not taking out the information that we actually want it to. Unsure how to fix this problem--may need to look into it further to see.





