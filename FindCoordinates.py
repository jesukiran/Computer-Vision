import cv2
import numpy as np
import sys
import os.path

# Usage: python sort_channel_coordinates.py <image_path>

def addPoint(p1, p2):
    return (p1[0] + p2[0], p1[1] + p2[1])
#Enter image as an argument from command prompt
def usage():
    print "python sort_channel_coordinates.py <channel_image_path>"

def main():

    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        usage()
        sys.exit(1)

    #Reading in the image as grayscale
    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

    #Copying the image
    mask = image.copy()

    current_point = image.shape

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            cPoint = (x, y)
            if image[x][y] != 0:
                if cv2.norm(cPoint) < cv2.norm(current_point):
                    current_point = cPoint

    # Path list!
    path = [current_point]

    #Invalidate used point
    image[current_point[0]][current_point[1]] = 0

    #Pseudocode as is
    while True:
        foundNeighbour = False
        for y in range(-1, 2):
            for x in range(-1, 2):
                if x != 0 or y != 0:
                    cPoint = addPoint(current_point, (x, y)) #Custom function to add points
                    if cPoint[1] < 0 or cPoint[1] >= image.shape[1] or cPoint[0] < 0 or cPoint[0] >= image.shape[0]:
                        continue

                    if image[cPoint[0]][cPoint[1]] != 0:
                        current_point = cPoint
                        path += [cPoint]
                        image[current_point[0]][current_point[1]] = 0
                        foundNeighbour = True

                    if foundNeighbour:
                        break
            if foundNeighbour:
                break

        if not foundNeighbour:
            break

    #Make a numpy matrix as a substitute for the Mat class in C++
    output = np.zeros(image.shape, dtype="uint8")
    nPoints = len(path)
    for i in range(nPoints):
        posRel = float(i) / nPoints
        val = int(255 * posRel)
        output[path[i][0]][path[i][1]] = val

    output = cv2.applyColorMap(output, cv2.COLORMAP_JET)

    #Invert and apply mask to color from blue to red!?
    newmask = 255 - mask
    output[newmask > 0] = 0


    cv2.imshow("output", output)
    cv2.waitKey(0)



main()