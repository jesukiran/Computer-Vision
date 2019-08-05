from __future__ import division
from __future__ import print_function


import cv2
import sys
import math
import gdspy
import logging
import operator

import numpy as np
from skimage.viewer.canvastools import RectangleTool
from skimage.viewer import ImageViewer
from skimage.morphology import skeletonize_3d
from scipy import spatial

import plotly
from plotly.graph_objs import Scatter


class SkeletonRegistration(object):

    def __init__(self):

        """
        Initialize parameter to store the skeleton coordinates

        """
        self.skel_coord  = []
        global viewer,coord_list

    def read_gds(self, gds_file, gds_layer):

        """
        Reads the user selected GDS mask.

        :param gds_file: Location of the GDS mask
        :type gds_file: string

        :param gds_layer: Name of the GDS layer ('PCM_TS', 'PCM_TD')
        :type gds_layer: string

        """

        self.gds_mask_name = gdspy.GdsLibrary()
        self.gds_mask_name.read_gds(gds_file)

        self.extract_gds_layer = self.gds_mask_name.extract(gds_layer)


        polys = self.extract_gds_layer.get_polygons(True)

        cnt_list = np.zeros((0,1,2))

        for chan in polys[inout_layer]:
            cnt = chan.reshape((-1,1,2)).astype('int32') // 2
            cnt_list = np.concatenate((cnt_list,cnt))

        minv = np.array(cnt_list).min(axis=0)
        maxv = np.array(cnt_list).max(axis=0)
        W,H = (maxv - minv).ravel()

        img = np.ones((int(H),int(W),3), dtype=np.uint8) * 255

        for chan in polys[inout_layer]:
            cnt = chan.reshape((-1,1,2)).astype('int32') // 2
            cnt -= minv.astype('int32')
            cv2.drawContours(img, [cnt], 0, (0,255,0), 3)

        # Add a border around
        top, bottom, left, right = [12]*4
        color = [255, 255, 255]
        img_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        # Convert the image to gray-scale
        self.gds_img = cv2.cvtColor(img_with_border, cv2.COLOR_BGR2GRAY)

        # Flip the image horizontally
        self.flipped_skeleton_img = cv2.flip(self.gds_img, 0)

        # Shape dimensions of the image
        w1, h1 = self.flipped_skeleton_img.shape[::-1]

        # Downsize the image by a factor of 2
        self.downsized_skeleton_img = cv2.pyrDown(self.flipped_skeleton_img, dstsize=(int(w1/2),int(h1/2)))

        # Perform thresholding
        self.skeleton_threshold_img = cv2.adaptiveThreshold(self.downsized_skeleton_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,7,2)

        # Use the 'close' morphological operator to fill up 1-pixel holes/join floating pixels
        kernel = np.ones((7,7),np.uint8)
        self.closing = cv2.morphologyEx(self.skeleton_threshold_img, cv2.MORPH_CLOSE, kernel)

        # Skeletonize
        self.skeleton_img = skeletonize_3d(self.closing)
        return self.skeleton_img


    def get_skeleton_coordinates(self, file_loc):

        """
        Store the skeleton coordinates in *Results/*.

        :param file_loc: Output file location where the skeleton coordinates need to be saved.
        :type file_loc: string
        """

        # Find every coordinate of the skeleton image
        _,cnts,_ = cv2.findContours(self.transformed_skeleton_img.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0]

        for c in cnts:
            x = c[0][0]
            y = c[0][1]

            self.skel_coord += [(x, y)]


        self.skel_coord = np.asarray(self.skel_coord)

        # Ensure unique coordinates only and no redundant coordinates
        index,_ = max(enumerate(self.skel_coord[:,0]), key = operator.itemgetter(1))
        skel_coord = self.skel_coord[0:index]

        # Visualize the skeleton coordinates (in red colour)
        img_skel = cv2.cvtColor(self.transformed_skeleton_img, cv2.COLOR_GRAY2BGR)
        for m in range(0, len(self.skel_coord)-1, 1):
            tmp_point = (self.skel_coord[m][0], self.skel_coord[m][1])
            cv2.circle(img_skel,tmp_point,1,(0,0,255),1)
        cv2.imwrite(file_loc+'Skeleton.png', img_skel)

        # Save the skeleton coordinates as a (.csv) file
        np.savetxt(file_loc+'Skeleton.csv', np.c_[skel_coord], delimiter = ',',fmt='%f')

        # Add the following information to the log file
        logging.info('Location of Skeleton Coordinates ---> ' + str(file_loc))

class FindMeniscusCoordinates(object):

    def __init__(self, mul_factor = 1, thresh_val = 0, skip_val = 1, tiff_file_start = 0, tiff_file_end = 0):

        """
        Initialize all parameters needed to store the meniscus coordinates.

        """

        self.actual_timestamp, self.actual_time  = ([] for i in range(2))
        self.indx, self.dT, self.coord = ([] for i in range(3))
        self.mul_factor = mul_factor
        self.thresh_val = thresh_val
        self.skip_val = skip_val
        self.tiff_file_start = tiff_file_start
        self.tiff_file_end = tiff_file_end

    def read_timestamp_data(self):

        """
        Reads timestamp information.
        """

        # Read the timestamp information
        data = np.genfromtxt(tiff_loc + 'timestamps.csv', dtype =None, delimiter = ",")
        self.actual_timestamp = np.asarray(data)

        return self.actual_timestamp

    def append_first_value(self, first_inlet_time_value):

        """
        Append the first timestamp value.

        :param first_inlet_time_value: Inlet time when liquid first enters the channel (normally, not seen in Image)
        :type first_inlet_time_value: int

        """
        # Read the skeleton coordinates
        temp_skel_coord = np.genfromtxt(file_loc+'Skeleton.csv', dtype =None, delimiter = ",")

        # Append the timestamp based on the index of the first coordinate
        first_val = temp_skel_coord[-1]
        self.coord +=[(int(first_val[0]),int(first_val[1]))]

        self.actual_time.append(first_inlet_time_value)
        return self.coord, self.actual_time

    def set_ROI(self):

        """
        Initialize the ROI coordinates. (X1,Y1) is the top-left corner of the ROI and (X2,Y2) is the bottom-right corner of the ROI.
        """
        self.X1 = 0
        self.X2 = 0

        self.Y1 = 0
        self.Y2 = 0

        return self.X1, self.X2, self.Y1, self.Y2

    def find_meniscus_coordinates(self):

        """
        Find all the meniscus points and their corresponding coordinates.
        """

        for i in range(self.tiff_file_start, self.tiff_file_end, 1):

            print('***INFO: Processing (.tiff) file ', end = '')

            # Read first image
            image_1 = cv2.imread(tiff_loc+ tfile%i, 0)

            # Based on the controls set, flip the image
            if right_control == 1:image_1 = cv2.flip(image_1, 1)
            if vertical_flip == 1: image_1 = cv2.flip(image_1, 0)

            # Create a ROI image
            image_1 = image_1[self.Y1:self.Y2, self.X1:self.X2]
            print ('---> ', tiff_loc+ tfile%i)

            # Read the consecutive image
            image_2 = cv2.imread(tiff_loc+ tfile%(i+1), 0)

            # Based on the controls set, flip the image
            if right_control == 1:image_2 = cv2.flip(image_2, 1)
            if vertical_flip == 1: image_2 = cv2.flip(image_2, 0)

            # Create a ROI image
            image_2 = image_2[self.Y1:self.Y2, self.X1:self.X2]

            # Take the difference between 2 images and multiply it by the  mul_factor to enhance the intensity.
            difference_image = cv2.subtract(image_2, image_1) * self.mul_factor

            # Take a threshold of the difference image
            _, threshold_image = cv2.threshold(difference_image, self.thresh_val, 255, cv2.THRESH_BINARY)

            # Similar to finding the coordinates of the skeleton, here we find the coordinates of the meniscus
            _,contours,_ = cv2.findContours(threshold_image.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0:

                c = max(contours, key = cv2.contourArea)
                (x,y),_ = cv2.minEnclosingCircle(c)
                _,_,w,h = cv2.boundingRect(c)
                area = w*h
                if int(area) > 1:

                    self.coord +=[(int(x),int(y))]
                    self.actual_time.append(int(self.actual_timestamp[i]))
                    self.indx.append(i)

        print('Done reading the image data.')

        # If you want to skip every few frames, change the skip_val. Default is 1.
        self.coord = self.coord[::self.skip_val]
        self.coord = np.asarray(self.coord)

        self.actual_time = self.actual_time[::self.skip_val]

        # If the lengths don't match, change the threshold values and rerun this script.
        if len(self.actual_time) != len(self.coord):
            raise ValueError('Check the coordinates AND timestamps.')

        return self.actual_time, self.coord

    def difference_between_timestamps(self):

        """
        Find the difference between consecutive timestamps.
        """

        for t in range(0, len(self.actual_time)-1, 1):

            # Unit conversion
            self.t1 = int(self.actual_time[t]) / 1000
            self.t2 = int(self.actual_time[t+1]) / 1000
            temp_diff = (self.t2 - self.t1) / 1000
            temp_diff = float(("%0.4f"%temp_diff))

            self.dT.append(temp_diff)
        # Save the time differences in a (.csv) file in Results/.
        np.savetxt(file_loc+'dT.csv', np.c_[self.dT], delimiter = ',',fmt='%f')
        return self.dT

    def visualize_meniscus_coordinates(self):

        """
        Visualize and save the meniscus coordinates.
        """

        # Save the meniscus coordinates in a (.csv) file in Results/.
        np.savetxt(file_loc+'Meniscus_Coordinates.csv', np.c_[self.coord], delimiter = ',',fmt='%f')

        # Visualize all the meniscus points
        meniscus_img = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)
        # Visualize the start meniscus point
        meniscus_start_img = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)
        # Visualize the end meniscus point
        meniscus_end_img = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)

        for m in range(0, len(self.coord)-1, 1):
            temp_point = (self.coord[m][0], self.coord[m][1])
            cv2.circle(meniscus_img,temp_point,3,(0,255,0),2)

        cv2.circle(meniscus_start_img, (int(self.coord[1][0]), int(self.coord[1][1])), 3, (0, 0, 255), 2)
        cv2.circle(meniscus_end_img, (int(self.coord[-1][0]), int(self.coord[-1][1])), 3, (0, 0, 255), 2)

        # Save the following in Results/.
        cv2.imwrite(file_loc + 'meniscus_coords_img.png', meniscus_img)
        cv2.imwrite(file_loc + 'meniscus_start_img.png', meniscus_start_img)
        cv2.imwrite(file_loc + 'meniscus_end_img.png', meniscus_end_img)

        np.savetxt(file_loc+'Timings.csv', np.c_[(self.actual_time[1]/1000, self.actual_time[-1]/1000)], delimiter = ',',fmt='%f')
        t_entry = self.actual_time[1]/1000
        t_exit = self.actual_time[-1]/1000

        return t_entry, t_exit

class PlotDistanceAndVelocity(object):

    def __init__(self, channel_length = 0.0):

        """
        Initialize parameters to find the distance and velocity.
        """

        self.dx = []
        self.dx_t = []
        self.channel_length = channel_length

    def read_coordinates(self):

        """
        Read meniscus and skeleton coordinates.
        """

        self.orig_meniscus_coordinates = np.genfromtxt(file_loc+'Meniscus_Coordinates.csv', dtype =None, delimiter = ",")
        self.skeleton_coordinates = np.genfromtxt(file_loc+'Skeleton.csv', dtype =None, delimiter = ",")
        self.skeleton_coordinates = np.asarray(self.skeleton_coordinates)

        self.orig_meniscus_coordinates = np.asarray(self.orig_meniscus_coordinates)

        # Read the difference of times
        dT = np.genfromtxt(file_loc+'dT.csv', dtype =float, delimiter = ",")

        self.T = [sum(dT[:i]) for i in range(1, len(dT)+1)]
        self.T = np.asarray(self.T)

        # Save time in the Results/
        np.savetxt(file_loc+'T.csv', np.c_[self.T], delimiter = ',',fmt='%f')

        return self.orig_meniscus_coordinates, self.skeleton_coordinates, self.T

    def get_dx(self):

        """
        Find the meniscus coordinates on the skeleton and calculate the length of the channel using the L2 distance formula.
        """

        tree = spatial.KDTree(self.skeleton_coordinates)
        indxpts = tree.query(self.orig_meniscus_coordinates)
        indxpts = np.asarray(indxpts)
        indxpts = indxpts[1,:]

        indxpts = np.asarray(indxpts)

        tmp_meniscus_coords = []

        # m1 indxpt is the first coordinate of the skeleton
        m1 = int(indxpts[0])
        # m2 indxpt is the first captured meniscus coordinate
        m2 = int(indxpts[1])
        # m3 indxpt is the last captured meniscus coordinate
        m3 = int(indxpts[-1])


        for j in range(m1, m2, -1):

            new_x = self.skeleton_coordinates[j][0]
            new_y = self.skeleton_coordinates[j][1]

            tmp_meniscus_coords += [(new_x, new_y)]


        tmp_distance = []


        for k in range(0, len(tmp_meniscus_coords)-2, 1):
            L1_distance = np.sqrt(((tmp_meniscus_coords[k+1][0]-tmp_meniscus_coords[k][0])**2) + ((tmp_meniscus_coords[k+1][1]-tmp_meniscus_coords[k][1])**2))
            tmp_distance.append(L1_distance)

        tmp_meniscus_coords = np.asarray(tmp_meniscus_coords)
        self.x_1 = (np.sum(tmp_distance)  * CALIBRATION) + ADD_EXTRA_CHANNEL_LENGTH
        np.savetxt(file_loc+'x_1.csv', np.c_[self.x_1], delimiter = ',',fmt='%f')
        np.savetxt(file_loc+'pt1.csv', np.c_[tmp_meniscus_coords], delimiter = ',',fmt='%f')


        tmp_meniscus_coords = []
        for j in range(m1, m3, -1):

            new_x = self.skeleton_coordinates[j][0]
            new_y = self.skeleton_coordinates[j][1]

            tmp_meniscus_coords += [(new_x, new_y)]


        tmp_distance = []


        for k in range(0, len(tmp_meniscus_coords)-2, 1):
            L2_distance = np.sqrt(((tmp_meniscus_coords[k+1][0]-tmp_meniscus_coords[k][0])**2) + ((tmp_meniscus_coords[k+1][1]-tmp_meniscus_coords[k][1])**2))
            tmp_distance.append(L2_distance)

        tmp_meniscus_coords = np.asarray(tmp_meniscus_coords)
        self.x_2 = (np.sum(tmp_distance)  * CALIBRATION) + ADD_EXTRA_CHANNEL_LENGTH
        np.savetxt(file_loc+'x_2.csv', np.c_[self.x_2], delimiter = ',',fmt='%f')
        np.savetxt(file_loc+'pt2.csv', np.c_[tmp_meniscus_coords], delimiter = ',',fmt='%f')



        for i in range(0, len(indxpts)-1 ,1):

            tmp_meniscus_coords_t = []
            m1_t = int(indxpts[0])

            m2_t = int(indxpts[i+1])


            for j in range(m1_t, m2_t, -1):

                new_x = self.skeleton_coordinates[j][0]
                new_y = self.skeleton_coordinates[j][1]

                tmp_meniscus_coords_t += [(new_x, new_y)]

            tmp_distance_t = []

            for k in range(0, len(tmp_meniscus_coords_t)-1, 1):
                L1_distance_t = np.sqrt(((tmp_meniscus_coords_t[k+1][0]-tmp_meniscus_coords_t[k][0])**2) + ((tmp_meniscus_coords_t[k+1][1]-tmp_meniscus_coords_t[k][1])**2))
                tmp_distance_t.append(L1_distance_t)

            temp_dist_t = np.sum(tmp_distance_t)
            self.dx_t.append(temp_dist_t)

        self.x = np.asarray(self.dx_t) * CALIBRATION
        self.x = self.x + ADD_EXTRA_CHANNEL_LENGTH

        np.savetxt(file_loc+'x.csv', np.c_[self.x], delimiter = ',',fmt='%f')
        print ("INFO: The length (in metres) of the channel is ---> "+str(self.x[-1]))
        print ("INFO: The added length (in metres) is ---> "+str(ADD_EXTRA_CHANNEL_LENGTH))

        for i in range(0, len(self.x)-1, 1):
            x = np.float(self.x[i+1]-self.x[i])
            self.dx.append(x)

        self.dx = np.asarray(self.dx)
        self.dx = np.insert(self.dx,0, self.x[0])

        np.savetxt(file_loc+'dx.csv', np.c_[self.dx], delimiter = ',',fmt='%f')

        if self.x[-1] >= self.channel_length:
            raise ValueError('Check the coordinates')

        return self.x_1, self.x_2, self.x, self.dx

    def plot_distance(self):

        """
        Plot Distance vs time. Result is located in *Results/Distance.html*.

        :param x: Distance travelled by the fluid in the channel
        :type x: int

        :param T: Time taken by the fluid in the channel
        :type T: int

        """

        trace = Scatter(
        x = self.T,
        y = self.x,


        mode = 'markers+lines',
        marker = dict(
                size = 10,
                color = 'rgba(172, 0, 0, .8)',
                line = dict(
                    width = 2,
                    color = 'rgb(0, 0, 0)'
                    )
                )
        )
        data = [trace]

        layout = dict(
                title = 'Fluid Distance travelled in Channel vs Time Taken',
                xaxis = dict(title = 'Time (in seconds) --->',titlefont=dict(family='Courier New, monospace', size=28, color='black'), tickfont=dict(family='Courier New, monospace', size=25, color='black')),
                yaxis = dict(title = 'Distance (in metres) --->',titlefont=dict(family='Courier New, monospace', size=28, color='black'), tickfont=dict(family='Courier New, monospace', size=25, color='black')),
                )

        fig = dict(data=data, layout=layout)

        np.savetxt(file_loc+'Distance.csv', np.c_[self.T,self.x], delimiter = ',', header = "Time, Distance",fmt='%.9f')
        plotly.offline.plot(fig, validate=False, filename=file_loc+'Distance.html', auto_open=False)

    def plot_velocity(self):

        """
        Plot velocity vs time. Result is located in *Results/Velocity.html*.

        :param velocity: Velocity of the fluid travelled in the channel
        :type velocity: int

        :param T: Time taken by the fluid in the channel
        :type T: int

        """

        velocity = self.x/self.T

        trace = Scatter(
        x = self.T,
        y = velocity,


        mode = 'markers+lines',
        marker = dict(
                size = 10,
                color = 'rgba(172, 0, 0, .8)',
                line = dict(
                    width = 2,
                    color = 'rgb(0, 90, 0)'
                    )
                )
        )
        data = [trace]

        layout = dict(
                title = 'Velocity of Fluid in Channel vs Time Taken',
                xaxis = dict(title = 'Time (in seconds) --->',titlefont=dict(family='Courier New, monospace', size=28, color='black'), tickfont=dict(family='Courier New, monospace', size=25, color='black')),
                yaxis = dict(title = 'Velocity (in m/s) --->',titlefont=dict(family='Courier New, monospace', size=28, color='black'), tickfont=dict(family='Courier New, monospace', size=25, color='black')),
                )

        fig = dict(data=data, layout=layout)

        np.savetxt(file_loc+'Velocity.csv', np.c_[self.T,velocity], delimiter = ',', header = "Time, Velocity",fmt='%f')
        plotly.offline.plot(fig, validate=False, filename=file_loc+'Velocity.html', auto_open=False)


class PlotContactAngle(object):

    def __init__(self, w = 0, h = 0, nL = 0.00095, nG = 18.6e-6, L = 0.0, gamma = 0.0722):

        """
        Initialize parameters
        """

        self.dx = []
        self.w = w
        self.h = h
        self.nL = nL
        self.nG = nG
        self.L = L
        self.gamma = gamma
        self.dump = []

    def read_parameters(self):

        """
        Read the parameters to calculate the contact angles.

        :param x_1: Distance till the first (entry) meniscus point
        :type x_1: float

        :param x_2: Distance till the last (exit) meniscus point
        :type x_2: float

        :param dT: Difference between timestamps
        :type dT: float

        :param x: Distance travelled by the fluid in the channel
        :type x: float

        :param dx: Distance between consecutive meniscus coordinates
        :type x: float

        :returns: x_1, x_2, dT, x, dx

        """

        self.x_1 = np.genfromtxt(file_loc+'x_1.csv', dtype =float, delimiter = ",")
        self.x_2 = np.genfromtxt(file_loc+'x_2.csv', dtype =float, delimiter = ",")

        self.dT = np.genfromtxt(file_loc+'dT.csv', dtype =float, delimiter = ",")
        self.x = np.genfromtxt(file_loc+'x.csv', dtype =float, delimiter = ",")
        self.dx = np.genfromtxt(file_loc+'dx.csv', dtype =float, delimiter = ",")

        return  self.x_1, self.x_2, self.dT, self.x, self.dx


    def calc_Rdash(self):

        """
        Calculate Rdash.
        """

        w = self.w
        h = self.h

        for n in range(1, 20, 1):
            eq1 = (((2*n)-1)*(math.pi)*w)/(2*h)
            eq2 = np.tanh(eq1)/(((2*n)-1)**5)
            self.dump.append(eq2)

        inside = ((192*h)/(w*((math.pi)**5))) * np.sum(self.dump)

        self.Rdash = (12/(w*(h**3)*(1-inside)))

        return self.Rdash


    def calc_contact_angle_between_inlet_outlet(self, t1, t2):

        """
        Calculate contact angle based on the entry meniscus point and the exit meniscus point.
        """


        # Unit conversion
        t1 = t1/1000
        t2 = t2/1000


        RL = (self.nL*self.Rdash)
        RG = (self.nG*self.Rdash)

        C = ((self.gamma) * ((2/self.h) + (2/self.w)))

        part1 = ((RL/2)*((self.x_2*self.x_2) - (self.x_1*self.x_1)))
        part2 = ((self.L*RG)*(self.x_2 - self.x_1))
        part3 = ((RG/2)*((self.x_2*self.x_2) - (self.x_1*self.x_1)))

        costheta = ((part1 + part2 - part3) * (self.h*self.w))/(C*(t2-t1))

        #np.minimum(1, costheta)
        theta = np.arccos(costheta)
        self.theta = np.rad2deg(theta)

        print("The new contact angle is: ", self.theta)

        return self.theta

    def calc_contact_angle_with_averaging(self):

        """
        Calculates the contact angle for every distance travelled by the meniscus in the channel.
        """

        averaged_velocity = []
        RL = self.x *(self.nL*self.Rdash)
        RG = (self.L-self.x)*(self.nG*self.Rdash)
        A = (self.w*self.h*(RL + RG)) / self.gamma
        B = np.sqrt(self.nL/self.gamma)

        velocity = self.dx/self.dT
        velocity = np.asarray(velocity)

        # A rolling-average is considered for velocity
        for q in range(2, len(velocity)-2, 1):
            tmp = (velocity[q-2] + velocity[q-1] + velocity[q] + velocity[q+1] + velocity[q+2])/5
            averaged_velocity.append(tmp)

        A = A[2:len(A)-2]
        part1 = (((A*averaged_velocity) +
                 (2*B*(np.sqrt(averaged_velocity)))) /
                 (1- (2*B*(np.sqrt(averaged_velocity)))))

        part2 = 1 / ((2/self.h) + (2/self.w))
        costheta = (part1 * part2)

        #np.minimum(1, costheta)
        theta = np.arccos(costheta)

        # Convert radians to degrees
        self.theta_in_deg = np.rad2deg(theta)

        self.x = self.x[2:len(self.x)-2]
        return self.theta_in_deg, self.x

    def plot_contact_angle(self):

        """
        Plot contact angle vs distance.

        :param x: Distance travelled by the fluid in the channel
        :type x: float

        :param theta_in_deg: Calculated contact angles
        :type theta_in_deg: float

        :returns: Contact angle plot is located in **Results/CA.html**.

        """

        trace = Scatter(
        x = self.x,
        y = self.theta_in_deg,


        mode = 'lines+markers',
        marker = dict(
                size = 10,
                color = 'rgba(172, 0, 0, .8)',
                line = dict(
                    width = 2,
                    color = 'rgb(0, 0, 123)'
                    )
                )
        )
        data = [trace]

        layout = dict(
                title = 'Contact Angle Plot',
                xaxis = dict(title = 'Distance (in meters) - - ->', titlefont=dict(family='Courier New, monospace', size=28, color='black'), tickfont=dict(family='Courier New, monospace', size=25, color='black')),
                yaxis = dict(title = 'Angle (in degrees) - - ->', titlefont=dict(family='Courier New, monospace', size=28, color='black'), nticks = 10, tick0 = 0 , range = [0, 90], tickfont=dict(family='Courier New, monospace', size=25, color='black'), autorange = False),
                )

        fig = dict(data=data, layout=layout)
        np.savetxt(file_loc +'CA.csv', np.c_[self.x, self.theta_in_deg], delimiter = ',', header = "Distance, Angle",fmt='%f')
        plotly.offline.plot(fig, validate=False, filename=file_loc +'CA.html', auto_open=False)


if __name__ == '__main__':

    tiff_loc = ''
    file_loc = ''
    reference_image = ''
    tfile = '%06d.tiff'
    CALIBRATION = ''
    ADD_EXTRA_CHANNEL_LENGTH = ''
    right_control = 0
    vertical_flip = 0

    inout_layer = (0, 0)
