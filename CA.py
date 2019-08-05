# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function



import os
import cv2
import wx
import TC
import csv
import glob
import shutil
import logging
import numpy as np
from skimage.morphology import skeletonize_3d


wildcard = "Python source (*.py)|*.py|" \
            "All files (*.*)|*.*"

class TimingChannelGUI(wx.Frame):

    def __init__(self, parent, title):

        """
        Initialization parameters for GUI

        """
        super(TimingChannelGUI, self).__init__(parent, title=title, size=(600, 400))

        self.InitUI()
        self.Centre()
        self.SetSize((700, 600))
        self.Show()
        self.skel_coord  = []
        self.field1 = []
        self.field2 = []
        self.field3 = []
        self.field4 = []
        self.field5 = []
        self.field6 = []
        self.field7 = []
        self.field8 = []


    def InitUI(self):

        """
        Build and setup the GUI

        """

        panel = wx.Panel(self)
        self.currentDirectory = os.getcwd()


        sizer = wx.GridBagSizer(5, 5)

        text1 = wx.StaticText(panel, label="Dataset to Process")
        sizer.Add(text1, pos=(1, 0), flag=wx.LEFT, border=10)

        self.tc1 = wx.TextCtrl(panel, wx.ID_ANY)
        sizer.Add(self.tc1, pos=(1, 1), span=(1, 1), flag=wx.TOP|wx.EXPAND)

        self.button1 = wx.Button(panel, wx.ID_ANY, label="Load Folder")
        self.button1.SetFocus()

        sizer.Add(self.button1, pos=(1, 2), flag=wx.LEFT, border=10)
        self.button1.Bind(wx.EVT_BUTTON, self.onLoadFolder)

        text7 = wx.StaticText(panel, label="Set Objective")
        sizer.Add(text7, pos=(2, 0), flag=wx.LEFT, border=10)

        self.combo7 = wx.ComboBox(panel, choices=['4x', '5x'], style = wx.CB_READONLY)
        self.combo7.Bind(wx.EVT_COMBOBOX, self.onSetObjective)
        sizer.Add(self.combo7, pos=(2, 1), span=(1, 1), flag=wx.TOP)

        self.cb0 = wx.CheckBox(panel, -1, 'Flip frame vertically', (10, 10))
        self.cb0.SetValue(True)

        self.cb0.Bind(wx.EVT_CHECKBOX, self.onFlipVertically)
        sizer.Add(self.cb0, pos=(2, 2), span=(1, 1), flag=wx.TOP)

        text6 = wx.StaticText(panel, label="Type of Mask")
        sizer.Add(text6, pos=(3, 0), flag=wx.LEFT, border=10)

        self.combo6 = wx.ComboBox(panel, choices=['SARNO_TS',
                                                  'SARNO_TD',
                                                  'SWIZZLE_TS_D02',
                                                  'SWIZZLE_TD_D02',
                                                  'SWIZZLE_TS_D03',
                                                  'SWIZZLE_TD_D03',
                                                  'SWIZZLE_TS_D04',
                                                  'SWIZZLE_TD_D04',
                                                  'SWIZZLE_TS_D05',
                                                  'SWIZZLE_TD_D05',
                                                  'SWIZZLE_TS_D06',
                                                  'SWIZZLE_TD_D06',
                                                  'SWIZZLE_TS_D08',
                                                  'SWIZZLE_TD_D08',
                                                  'SWIZZLE_TS_D09',
                                                  'SWIZZLE_TD_D09'], style = wx.CB_READONLY)
        self.combo6.Bind(wx.EVT_COMBOBOX, self.onSetMask)
        sizer.Add(self.combo6, pos=(3, 1), span=(1, 1), flag=wx.TOP)

        self.cb = wx.CheckBox(panel, -1, 'Flip frame horizontally', (10, 10))
        self.cb.SetValue(True)

        self.cb.Bind(wx.EVT_CHECKBOX, self.onFlipHorizontally)
        sizer.Add(self.cb, pos=(3, 2), span=(1, 1), flag=wx.TOP)

        line4 = wx.StaticLine(panel)
        sizer.Add(line4, pos=(4, 0), span=(1, 20), flag=wx.EXPAND|wx.BOTTOM, border=10)

        text_xtrans = wx.StaticText(panel, label="Translation (X-axis)")
        sizer.Add(text_xtrans, pos=(5, 2), flag=wx.LEFT, border=10)

        self.scx = wx.SpinCtrl(panel, wx.ID_ANY,'',(10, -1))
        self.scx.SetRange(-400, 400)
        self.scx.SetValue(0)
        self.scx.Bind(wx.EVT_SPINCTRL, self.onSpin_xaxis)
        sizer.Add(self.scx, pos=(5, 3), flag=wx.LEFT)

        text_ytrans = wx.StaticText(panel, label="Translation (Y-axis)")
        sizer.Add(text_ytrans, pos=(5, 4), flag=wx.LEFT, border=10)

        self.scy = wx.SpinCtrl(panel, wx.ID_ANY,'',(10, -1))
        self.scy.SetRange(-400, 400)
        self.scy.SetValue(0)
        self.scy.Bind(wx.EVT_SPINCTRL, self.onSpin_yaxis)
        sizer.Add(self.scy, pos=(5, 5), flag=wx.LEFT)

        img = wx.Image(640, 512)
        self.imageCtrl = wx.StaticBitmap(panel, wx.ID_ANY, wx.Bitmap(img))
        sizer.Add(self.imageCtrl, pos=(6, 0), span=(1, 2), flag=wx.LEFT, border=10)

        self.button4 = wx.Button(panel, wx.ID_ANY, label="Register Skeleton")
        self.button4.SetFocus()
        sizer.Add(self.button4, pos=(6, 2), flag=wx.LEFT, border=10)
        self.button4.Bind(wx.EVT_BUTTON, self.onRegister)

        text9 = wx.StaticText(panel, label="Threshold Value")
        sizer.Add(text9, pos=(8, 0), flag=wx.LEFT, border=10)

        self.tc9 = wx.TextCtrl(panel)
        self.tc9.Bind(wx.EVT_TEXT, self.onThresh)
        sizer.Add(self.tc9, pos=(8, 1), span=(1, 1), flag=wx.RIGHT)

        self.button8 = wx.Button(panel, wx.ID_ANY, label="Visualize Coordinates")
        self.button8.SetFocus()
        sizer.Add(self.button8, pos=(9, 0), flag=wx.LEFT, border=10)
        self.button8.Bind(wx.EVT_BUTTON, self.onVisualize_coordinates)

        line3 = wx.StaticLine(panel)
        sizer.Add(line3, pos=(10, 0), span=(1, 20), flag=wx.EXPAND|wx.BOTTOM, border=10)


        self.button9 = wx.Button(panel, wx.ID_ANY, label="Run")
        self.button9.SetFocus()
        sizer.Add(self.button9, pos=(11, 0), flag=wx.LEFT, border=10)
        self.button9.Bind(wx.EVT_BUTTON, self.onRun)

        self.button10 = wx.Button(panel, wx.ID_ANY, label="Stop")
        self.button10.SetFocus()
        sizer.Add(self.button10, pos=(11, 1), flag=wx.LEFT, border=10)
        self.button10.Bind(wx.EVT_BUTTON, self.onStop)

        panel.SetSizer(sizer)


    def onLoadFolder(self, event):

        """
        Loads the chosen folder.

        :param event: Loads the folder path
        :type event: event

        """


        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.currentDirectory,
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
            )
        if dlg.ShowModal() == wx.ID_OK:
            filepath = dlg.GetPath()
            self.folderpath = filepath + ('/')
            os.chdir('../.')
            self.folderpath = os.getcwd().replace('\\', '/') + ('/')
            self.tc1.SetValue(self.folderpath)
            logging.basicConfig(filename=self.folderpath+"logfilename.log", level=logging.INFO)
            logging.info('The file path is:' + str(self.folderpath))

        self.get_firstdir()
        dlg.Destroy()


    def get_firstdir(self):

        """
        Loads the a folder for pre-processing.
        """


        folders = next(os.walk(self.folderpath))[1]

        self.refpath = self.folderpath + folders[0] +'/'
        os.chdir(self.refpath)

        for files in glob.glob("*.tiff"):
                f = os.path.splitext(files)[0]

        self.file_length = len(f)
        self.refpath_res = self.refpath + 'Results/'
        if not os.path.exists(self.refpath_res):
            os.makedirs(self.refpath_res)
        else:
            shutil.rmtree(self.refpath_res)
            os.makedirs(self.refpath_res)


        self.start_val = 10
        self.end_val = len(glob.glob1(self.refpath,"*.tiff")) - 1
        self.setReferenceImage(self.refpath)

    def get_subdirs(self, event):

        """
        Loads a folder and calculates the contact angles.

        :param event: Loads the folder path
        :type event: event

        """


        folders = next(os.walk(self.folderpath))[1]

        fol_cnt = len(folders)
        for i in range(0, fol_cnt, 1):
            self.filepath = self.folderpath + folders[i] + '/'
            die_name = folders[i][-5:]
            os.chdir(self.filepath)
            for files in glob.glob("*.tiff"):
                f = os.path.splitext(files)[0]
            self.file_length = len(f)
            TC.file_loc = self.filepath + 'Results/'

            if not os.path.exists(TC.file_loc):
                os.makedirs(TC.file_loc)
            else:
                shutil.rmtree(TC.file_loc)
                os.makedirs(TC.file_loc)

            self.start_val = 100
            self.end_val = len(glob.glob1(self.filepath,"*.tiff")) - 1
            self.setReferenceImage(self.filepath)

            self.translation_rotation_matrix = np.float32([[self.scale_factor, 0, self.xaxis],[0, self.scale_factor, self.yaxis]])
            self.transformed_skeleton_img = cv2.warpAffine(self.skeleton_img, self.translation_rotation_matrix, (self.w_ref,self.h_ref))

            kernel = np.ones((7,7),np.uint8)
            closing = cv2.morphologyEx(self.transformed_skeleton_img.astype('uint8'), cv2.MORPH_CLOSE, kernel)
            self.transformed_skeleton_img = skeletonize_3d(closing)

            self.overlay_image = cv2.addWeighted(self.reference_image, 0.7, self.transformed_skeleton_img, 0.4, 0)

            cv2.imwrite(TC.file_loc + 'registered_skeleton.png', self.overlay_image)

            obj = TC.SkeletonRegistration()
            obj.transformed_skeleton_img = self.transformed_skeleton_img
            obj.get_skeleton_coordinates(TC.file_loc)

            try:
                self.findMeniscus(self.filepath, event)
                self.onRun(self.filepath)
            except:

                self.ca_angle = str('NA')
                self.x_1 = str('NA')
                self.x_2 = str('NA')


            try:
                temp_timing = np.genfromtxt(TC.file_loc + 'Timings.csv', dtype = float, delimiter = ",")
                temp_time = (temp_timing[1]/1000) - (temp_timing[0]/1000)
                t1 = temp_timing[0]/1000
                t2 = temp_timing[1]/1000
            except:
                temp_time = str('NA')
                t1 = str('NA')
                t2 = str('NA')

                self.ca_angle = str('NA')
                self.x_1 = str('NA')
                self.x_2 = str('NA')


            self.field1 += [die_name]
            self.field2 += [t1]
            self.field3 += [t2]

            self.field4 += [temp_time]
            self.field5 += [self.x_1]
            self.field6 += [self.x_2]
            self.field7 += [self.ca_angle]
            self.field8 += [str(TC.file_loc)]

        self.field1 = np.asarray(self.field1)
        self.field2 = np.asarray(self.field2)
        self.field3 = np.asarray(self.field3)
        self.field4 = np.asarray(self.field4)

        self.field5 = np.asarray(self.field5)
        self.field6 = np.asarray(self.field6)
        self.field7 = np.asarray(self.field7)
        self.field8 = np.asarray(self.field8)

        outfile = (self.field1, self.field2, self.field3, self.field4, self.field5, self.field6, self.field7, self.field8)
        np.savetxt(self.folderpath + 'Report.csv', np.c_[outfile], delimiter = ',',header = 'Die Number, T1, T2, T2-T1, X1, X2, CA, Path', fmt='%s')

    def onFlipHorizontally(self, event):

        """
        When the box is checked, the flag is set to 1; to set flip the frame horizontally.

        :param event: Flip Horizontally
        :type event: event

        :returns: Flag is set to 1 when checked, else the flag is 0
        """

        if self.cb.GetValue():
            self.right_control = 1
            TC.right_control = 1

        else:
            self.right_control = 0
            TC.right_control = 0

    def onFlipVertically(self, event):

        """
        When the box is checked, the flag is set to 1; to set flip the frame vertically.

        :param event: Flip Vertically
        :type event: event

        :returns: Flag is set to 1 when checked, else the flag is 0
        """

        if self.cb0.GetValue():
            self.vertical_flip = 1
            TC.vertical_flip = 1

        else:
            self.vertical_flip = 0
            TC.vertical_flip = 0

    def onSetObjective(self, event):

        """
        When the box is checked, the flag is set to 1; to set flip the frame vertically.

        :param event: Flip Vertically
        :type event: event

        :returns: Flag is set to 1 when checked, else the flag is 0
        """

        self.calib_val = str(self.combo7.GetValue())

        return self.calib_val

    def onSetMask(self, event):

        """
        Reads the GDS mask and gets the parameters
        """

        self.channel_mask = str(self.combo6.GetValue())
        self.get_parameters()
        self.create_skeleton(event)

        return self.channel_mask

    def setReferenceImage(self, path):

        """
        Reads a reference image for pre-processing
        """

        k = self.start_val - 3

        if self.file_length == 6:
            TC.tfile = '%06d.tiff'

            self.filename = path + ('%06d.tiff'%k)
            logging.info("The reference image is: " + str(self.filename))


        elif self.file_length == 5:
            TC.tfile = '%05d.tiff'

            self.filename = path + ('%05d.tiff'%k)
            logging.info("The reference image is: " + str(self.filename))

        else:

            raise ValueError('Check the length of filenames.')
            logging.error('Check the length of filenames.')

    def get_parameters(self):

        """
        Reads the parameters for the selected mask
        """

        mask_param_loc = '/imec/windows/milab01/WP3 LFI/ufluidics/Automation/Software/CA_Analysis_Tool/Parameters/'

        param = np.genfromtxt(mask_param_loc + 'CA_Parameters.csv', dtype = None, delimiter = ",")

        self.nL = np.float(param[0][1])
        self.nG = np.float(param[1][1])
        self.gamma = np.float(param[2][1])
        self.mul_factor = int(param[3][1])
        self.skip_val = int(param[4][1])


        if self.calib_val == '4x':

            val = np.genfromtxt(mask_param_loc + 'Scale.csv', dtype = None, delimiter = ",", encoding=None).tolist()
            self.scale_factor = np.float(val[1])
            TC.CALIBRATION = 5e-6

            print ('***INFO: Magnification and Scale Factor: ', self.calib_val, self.scale_factor)

        elif self.calib_val == '5x':

            self.scale_factor = 1
            TC.CALIBRATION = 4e-6

            print ('***INFO: Magnification and Scale Factor: ', self.calib_val, self.scale_factor)

        else:

            raise ValueError('Enter valid magnification factor.')


        print ('***INFO: nL, nG, gamma, mul_factor, skip_val:', self.nL, self.nG, self.gamma, self.mul_factor, self.skip_val)

        if self.channel_mask == 'SARNO_TS':

            self.GDS_Layer = 'PCM_TS'
            TC.inout_layer = (900, 0)

            data = np.genfromtxt(mask_param_loc + 'SARNO_TS.csv', dtype = None, delimiter = ",", encoding=None)
            self.GDS_filename = str(data[0][1])
            self.channel_length = np.float(data[1][1])
            self.channel_width = np.float(data[2][1])
            self.channel_height = np.float(data[3][1])
            TC.ADD_EXTRA_CHANNEL_LENGTH = np.float(data[4][1])

            print ('***INFO: GDS File Selected, Channel Height, Channel Length, Channel Width: ', self.GDS_filename, self.channel_height, self.channel_length, self.channel_width)


        elif self.channel_mask == 'SARNO_TD':

            self.GDS_Layer = 'PCM_TD'
            TC.inout_layer = (910, 0)

            data = np.genfromtxt(mask_param_loc + 'SARNO_TD.csv', dtype = None, delimiter = ",", encoding=None)

            self.GDS_filename = str(data[0][1])
            self.channel_length = np.float(data[1][1])
            self.channel_width = np.float(data[2][1])
            self.channel_height = np.float(data[3][1])
            TC.ADD_EXTRA_CHANNEL_LENGTH = np.float(data[4][1])

            print ('***INFO: GDS File Selected, Channel Height, Channel Length, Channel Width: ', self.GDS_filename, self.channel_height, self.channel_length, self.channel_width)


        elif self.channel_mask == 'SWIZZLE_TS_D02':

            self.GDS_Layer = 'PCM_TS'
            TC.inout_layer = (920, 0)

            data = np.genfromtxt(mask_param_loc + 'SWIZZLE_TS_D02.csv', dtype = None, delimiter = ",",  encoding=None)
            self.GDS_filename = str(data[0][1])
            self.channel_length = np.float(data[1][1])
            self.channel_width = np.float(data[2][1])
            self.channel_height = np.float(data[3][1])
            TC.ADD_EXTRA_CHANNEL_LENGTH = np.float(data[4][1])

            print ('***INFO: GDS File Selected, Channel Height, Channel Length, Channel Width: ', self.GDS_filename, self.channel_height, self.channel_length, self.channel_width)

        elif self.channel_mask == 'SWIZZLE_TD_D02':

            self.GDS_Layer = 'PCM_TD'
            TC.inout_layer = (930, 0)

            data = np.genfromtxt(mask_param_loc + 'SWIZZLE_TD_D02.csv', dtype = None, delimiter = ",",  encoding=None)
            self.GDS_filename = str(data[0][1])
            self.channel_length = np.float(data[1][1])
            self.channel_width = np.float(data[2][1])
            self.channel_height = np.float(data[3][1])
            TC.ADD_EXTRA_CHANNEL_LENGTH = np.float(data[4][1])

            print ('***INFO: GDS File Selected, Channel Height, Channel Length, Channel Width: ', self.GDS_filename, self.channel_height, self.channel_length, self.channel_width)

        elif self.channel_mask == 'SWIZZLE_TS_D03':

            self.GDS_Layer = 'PCM_TS'
            TC.inout_layer = (920, 0)

            data = np.genfromtxt(mask_param_loc + 'SWIZZLE_TS_D03.csv', dtype = None, delimiter = ",",  encoding=None)
            self.GDS_filename = str(data[0][1])
            self.channel_length = np.float(data[1][1])
            self.channel_width = np.float(data[2][1])
            self.channel_height = np.float(data[3][1])
            TC.ADD_EXTRA_CHANNEL_LENGTH = np.float(data[4][1])

            print ('***INFO: GDS File Selected, Channel Height, Channel Length, Channel Width: ', self.GDS_filename, self.channel_height, self.channel_length, self.channel_width)


        elif self.channel_mask == 'SWIZZLE_TD_D03':

            self.GDS_Layer = 'PCM_TD'
            TC.inout_layer = (930, 0)

            data = np.genfromtxt(mask_param_loc + 'SWIZZLE_TD_D03.csv', dtype = None, delimiter = ",",  encoding=None)
            self.GDS_filename = str(data[0][1])
            self.channel_length = np.float(data[1][1])
            self.channel_width = np.float(data[2][1])
            self.channel_height = np.float(data[3][1])
            TC.ADD_EXTRA_CHANNEL_LENGTH = np.float(data[4][1])

            print ('***INFO: GDS File Selected, Channel Height, Channel Length, Channel Width: ', self.GDS_filename, self.channel_height, self.channel_length, self.channel_width)


        elif self.channel_mask == 'SWIZZLE_TS_D04':

            self.GDS_Layer = 'PCM_TS'
            TC.inout_layer = (920, 0)

            data = np.genfromtxt(mask_param_loc + 'SWIZZLE_TS_D04.csv', dtype = None, delimiter = ",",  encoding=None)
            self.GDS_filename = str(data[0][1])
            self.channel_length = np.float(data[1][1])
            self.channel_width = np.float(data[2][1])
            self.channel_height = np.float(data[3][1])
            TC.ADD_EXTRA_CHANNEL_LENGTH = np.float(data[4][1])

            print ('***INFO: GDS File Selected, Channel Height, Channel Length, Channel Width: ', self.GDS_filename, self.channel_height, self.channel_length, self.channel_width)


        elif self.channel_mask == 'SWIZZLE_TD_D04':

            self.GDS_Layer = 'PCM_TD'
            TC.inout_layer = (930, 0)

            data = np.genfromtxt(mask_param_loc + 'SWIZZLE_TD_D04.csv', dtype = None, delimiter = ",",  encoding=None)
            self.GDS_filename = str(data[0][1])
            self.channel_length = np.float(data[1][1])
            self.channel_width = np.float(data[2][1])
            self.channel_height = np.float(data[3][1])
            TC.ADD_EXTRA_CHANNEL_LENGTH = np.float(data[4][1])

            print ('***INFO: GDS File Selected, Channel Height, Channel Length, Channel Width: ', self.GDS_filename, self.channel_height, self.channel_length, self.channel_width)

        elif self.channel_mask == 'SWIZZLE_TS_D05':

            self.GDS_Layer = 'PCM_TS'
            TC.inout_layer = (920, 0)

            data = np.genfromtxt(mask_param_loc + 'SWIZZLE_TS_D05.csv', dtype = None, delimiter = ",",  encoding=None)
            self.GDS_filename = str(data[0][1])
            self.channel_length = np.float(data[1][1])
            self.channel_width = np.float(data[2][1])
            self.channel_height = np.float(data[3][1])
            TC.ADD_EXTRA_CHANNEL_LENGTH = np.float(data[4][1])

            print ('***INFO: GDS File Selected, Channel Height, Channel Length, Channel Width: ', self.GDS_filename, self.channel_height, self.channel_length, self.channel_width)


        elif self.channel_mask == 'SWIZZLE_TD_D05':

            self.GDS_Layer = 'PCM_TD'
            TC.inout_layer = (930, 0)

            data = np.genfromtxt(mask_param_loc + 'SWIZZLE_TD_D05.csv', dtype = None, delimiter = ",",  encoding=None)
            self.GDS_filename = str(data[0][1])
            self.channel_length = np.float(data[1][1])
            self.channel_width = np.float(data[2][1])
            self.channel_height = np.float(data[3][1])
            TC.ADD_EXTRA_CHANNEL_LENGTH = np.float(data[4][1])

            print ('***INFO: GDS File Selected, Channel Height, Channel Length, Channel Width: ', self.GDS_filename, self.channel_height, self.channel_length, self.channel_width)

        elif self.channel_mask == 'SWIZZLE_TS_D06':

            self.GDS_Layer = 'PCM_TS'
            TC.inout_layer = (920, 0)

            data = np.genfromtxt(mask_param_loc + 'SWIZZLE_TS_D06.csv', dtype = None, delimiter = ",",  encoding=None)
            self.GDS_filename = str(data[0][1])
            self.channel_length = np.float(data[1][1])
            self.channel_width = np.float(data[2][1])
            self.channel_height = np.float(data[3][1])
            TC.ADD_EXTRA_CHANNEL_LENGTH = np.float(data[4][1])

            print ('***INFO: GDS File Selected, Channel Height, Channel Length, Channel Width: ', self.GDS_filename, self.channel_height, self.channel_length, self.channel_width)


        elif self.channel_mask == 'SWIZZLE_TD_D06':

            self.GDS_Layer = 'PCM_TD'
            TC.inout_layer = (930, 0)

            data = np.genfromtxt(mask_param_loc + 'SWIZZLE_TD_D06.csv', dtype = None, delimiter = ",",  encoding=None)
            self.GDS_filename = str(data[0][1])
            self.channel_length = np.float(data[1][1])
            self.channel_width = np.float(data[2][1])
            self.channel_height = np.float(data[3][1])
            TC.ADD_EXTRA_CHANNEL_LENGTH = np.float(data[4][1])

            print ('***INFO: GDS File Selected, Channel Height, Channel Length, Channel Width: ', self.GDS_filename, self.channel_height, self.channel_length, self.channel_width)

        elif self.channel_mask == 'SWIZZLE_TS_D08':

            self.GDS_Layer = 'PCM_TS'
            TC.inout_layer = (920, 0)

            data = np.genfromtxt(mask_param_loc + 'SWIZZLE_TS_D08.csv', dtype = None, delimiter = ",",  encoding=None)
            self.GDS_filename = str(data[0][1])
            self.channel_length = np.float(data[1][1])
            self.channel_width = np.float(data[2][1])
            self.channel_height = np.float(data[3][1])
            TC.ADD_EXTRA_CHANNEL_LENGTH = np.float(data[4][1])

            print ('***INFO: GDS File Selected, Channel Height, Channel Length, Channel Width: ', self.GDS_filename, self.channel_height, self.channel_length, self.channel_width)


        elif self.channel_mask == 'SWIZZLE_TD_D08':

            self.GDS_Layer = 'PCM_TD'
            TC.inout_layer = (930, 0)

            data = np.genfromtxt(mask_param_loc + 'SWIZZLE_TD_D08.csv', dtype = None, delimiter = ",",  encoding=None)
            self.GDS_filename = str(data[0][1])
            self.channel_length = np.float(data[1][1])
            self.channel_width = np.float(data[2][1])
            self.channel_height = np.float(data[3][1])
            TC.ADD_EXTRA_CHANNEL_LENGTH = np.float(data[4][1])

            print ('***INFO: GDS File Selected, Channel Height, Channel Length, Channel Width: ', self.GDS_filename, self.channel_height, self.channel_length, self.channel_width)


        elif self.channel_mask == 'SWIZZLE_TS_D09':

            self.GDS_Layer = 'PCM_TS'
            TC.inout_layer = (920, 0)

            data = np.genfromtxt(mask_param_loc + 'SWIZZLE_TS_D09.csv', dtype = None, delimiter = ",",  encoding=None)
            self.GDS_filename = str(data[0][1])
            self.channel_length = np.float(data[1][1])
            self.channel_width = np.float(data[2][1])
            self.channel_height = np.float(data[3][1])
            TC.ADD_EXTRA_CHANNEL_LENGTH = np.float(data[4][1])

            print ('***INFO: GDS File Selected, Channel Height, Channel Length, Channel Width: ', self.GDS_filename, self.channel_height, self.channel_length, self.channel_width)


        elif self.channel_mask == 'SWIZZLE_TD_D09':

            self.GDS_Layer = 'PCM_TD'
            TC.inout_layer = (930, 0)

            data = np.genfromtxt(mask_param_loc + 'SWIZZLE_TD_D09.csv', dtype = None, delimiter = ",",  encoding=None)
            self.GDS_filename = str(data[0][1])
            self.channel_length = np.float(data[1][1])
            self.channel_width = np.float(data[2][1])
            self.channel_height = np.float(data[3][1])
            TC.ADD_EXTRA_CHANNEL_LENGTH = np.float(data[4][1])

            print ('***INFO: GDS File Selected, Channel Height, Channel Length, Channel Width: ', self.GDS_filename, self.channel_height, self.channel_length, self.channel_width)

        else:
            raise ValueError('Mask is not valid.')


    def onThresh(self, event):

        """
        Set the entered threshold value

        :param event: Sets the threshold value
        :type event: event

        :returns: Threshold value
        """
        self.thresh_val = int(event.GetString())

        return self.thresh_val


    def onSpin_xaxis(self, event):
        """
        Reads the user set translation value along x-axis

        :param event: Sets the translation value along x-axis
        :type event: event

        :returns: Translation value along x-axis
        """
        self.xaxis = int(self.scx.GetValue())
        return self.xaxis

    def onSpin_yaxis(self, event):

        """
        Reads the user set translation value along y-axis

        :param event: Sets the translation value along y-axis
        :type event: event

        :returns: Translation value along y-axis
        """
        self.yaxis = int(self.scy.GetValue())
        return self.yaxis


    def onRegister(self, event):

        """
        Registers the skeleton on the reference image

        :param event: Registers the skeleton
        :type event: event

        :returns: Using the user set translation values along x- and y- axis, the skeleton is registered onto the reference image.
        """


        TC.file_loc = self.refpath + 'Results/'
        self.translation_rotation_matrix = np.float32([[self.scale_factor, 0, self.xaxis],[0, self.scale_factor, self.yaxis]])
        self.transformed_skeleton_img = cv2.warpAffine(self.skeleton_img, self.translation_rotation_matrix, (self.w_ref,self.h_ref))



        kernel = np.ones((7,7),np.uint8)
        closing = cv2.morphologyEx(self.transformed_skeleton_img.astype('uint8'), cv2.MORPH_CLOSE, kernel)
        self.transformed_skeleton_img = skeletonize_3d(closing)

        self.overlay_image = cv2.addWeighted(self.reference_image, 0.7, self.transformed_skeleton_img, 0.4, 0)

        cv2.imwrite(TC.file_loc + 'registered_skeleton.png', self.overlay_image)
        reg_loc = TC.file_loc + 'registered_skeleton.png'
        img = wx.Image(reg_loc, wx.BITMAP_TYPE_ANY)
        self.imageCtrl.SetBitmap(wx.Bitmap(img))

        obj = TC.SkeletonRegistration()
        obj.transformed_skeleton_img = self.transformed_skeleton_img
        obj.get_skeleton_coordinates(TC.file_loc)


    def create_skeleton(self, event):

        """
        Creates the skeleton for the selected mask.

        """

        reference_image = cv2.imread(self.filename, 0)

        self.onFlipHorizontally(event)
        self.onFlipVertically(event)

        if self.right_control == 1:reference_image = cv2.flip(reference_image, 1)
        if self.vertical_flip == 1: reference_image = cv2.flip(reference_image, 0)
        self.reference_image = reference_image

        self.w_ref, self.h_ref = self.reference_image.shape[::-1]

        TC.gds_file_loc = self.GDS_filename
        TC.gds_layer = self.GDS_Layer

        obj_skel = TC.SkeletonRegistration()

        skeleton_img = obj_skel.read_gds(gds_file = self.GDS_filename, gds_layer = self.GDS_Layer)
        print(skeleton_img.shape)
        if self.channel_mask == 'SARNO_TS':
            self.skeleton_img = skeleton_img[0:553, 80:950]

        elif self.channel_mask == 'SARNO_TD':
            self.skeleton_img = skeleton_img[0:624, 80:1000]

        elif self.channel_mask == 'SWIZZLE_TS_D02':
            self.skeleton_img = skeleton_img[0:464, 100:780]

        elif self.channel_mask == 'SWIZZLE_TD_D02':
            self.skeleton_img = skeleton_img[0:464, 100:780]

        elif self.channel_mask == 'SWIZZLE_TS_D03':
            self.skeleton_img = skeleton_img[0:553, 100:930]

        elif self.channel_mask == 'SWIZZLE_TD_D03':
            self.skeleton_img = skeleton_img[0:553, 100:930]

        elif self.channel_mask == 'SWIZZLE_TS_D04':
            self.skeleton_img = skeleton_img[0:553, 100:930]

        elif self.channel_mask == 'SWIZZLE_TD_D04':
            self.skeleton_img = skeleton_img[0:553, 100:930]

        elif self.channel_mask == 'SWIZZLE_TS_D05':
            self.skeleton_img = skeleton_img[0:553, 100:930]

        elif self.channel_mask == 'SWIZZLE_TD_D05':
            self.skeleton_img = skeleton_img[0:553, 100:930]

        elif self.channel_mask == 'SWIZZLE_TS_D06':
            self.skeleton_img = skeleton_img[0:553, 100:930]

        elif self.channel_mask == 'SWIZZLE_TD_D06':
            self.skeleton_img = skeleton_img[0:553, 100:930]

        elif self.channel_mask == 'SWIZZLE_TS_D08':
            self.skeleton_img = skeleton_img[0:553, 100:930]

        elif self.channel_mask == 'SWIZZLE_TD_D08':
            self.skeleton_img = skeleton_img[0:553, 100:930]

        elif self.channel_mask == 'SWIZZLE_TS_D09':
            self.skeleton_img = skeleton_img[0:553, 100:930]

        elif self.channel_mask == 'SWIZZLE_TD_D09':
            self.skeleton_img = skeleton_img[0:553, 100:930]

        else:
            raise ValueError('Mask not found.')

        return self.skeleton_img

    def findMeniscus(self, path, event):

        """
        Find the meniscus coordinates

        """

        TC.tiff_loc = path
        TC.file_loc = path + 'Results/'
        TC.reference_image = self.reference_image

        with open(TC.tiff_loc + 'timestamps.txt') as inf:
            reader = csv.reader(inf, delimiter="\t")
            second_col = list(zip(*reader))[1]
            second_col = np.asarray(second_col).astype('int')


        np.savetxt(TC.tiff_loc +'timestamps.csv', np.c_[second_col], fmt='%f')


        obj_vis = TC.FindMeniscusCoordinates(mul_factor = self.mul_factor,
                                            thresh_val = self.thresh_val,
                                            skip_val = self.skip_val,
                                            tiff_file_start = self.start_val,
                                            tiff_file_end = self.end_val)

        obj_vis.mul_factor = self.mul_factor
        obj_vis.tiff_file_start = self.start_val
        obj_vis.tiff_file_end = self.end_val
        obj_vis.thresh_val = self.thresh_val
        obj_vis.skip_val = self.skip_val
        obj_vis.X1 = 0
        obj_vis.Y1 = 0
        obj_vis.X2 = self.h_ref
        obj_vis.Y2 = self.w_ref
        inlet_time = second_col[self.start_val-1]

        obj_vis.read_timestamp_data()
        obj_vis.append_first_value(inlet_time)

        self.onFlipHorizontally(event)
        self.onFlipVertically(event)

        obj_vis.find_meniscus_coordinates()

        obj_vis.difference_between_timestamps()

        self.t_entry, self.t_exit = obj_vis.visualize_meniscus_coordinates()

    def onVisualize_coordinates(self, event):

        """
        Visualize coordinates for visual check.

        """

        try:
            self.findMeniscus(self.refpath, event)
            men_img = self.refpath + 'Results/' + 'meniscus_coords_img.png'
            img = wx.Image(men_img, wx.BITMAP_TYPE_ANY)
            self.imageCtrl.SetBitmap(wx.Bitmap(img))

            selecting = True
            if wx.MessageBox("Is this threshold OK? Proceed and run the script?","Final Check . . .",wx.YES_NO) == wx.YES:
                self.get_subdirs(event)
            else:
                selecting = False
        except:
            self.get_subdirs(event)


    def onRun(self, path):

        """
        Run script to calculate the distance, velocity and contact angle.

        """

        TC.tiff_loc = path
        TC.file_loc = path + 'Results/'

        TC.reference_image = cv2.imread(self.filename, 0)
        TC.reference_image = cv2.flip(TC.reference_image, 1)

        obj_pdv = TC.PlotDistanceAndVelocity(self.channel_length)
        obj_pdv.read_coordinates()
        obj_pdv.get_dx()
        obj_pdv.plot_distance()
        obj_pdv.plot_velocity()
        obj_pcona = TC.PlotContactAngle(w = self.channel_width,
                                            h = self.channel_height,
                                            nL = self.nL, nG = self.nG,
                                            L = self.channel_length,
                                            gamma = self.gamma)
        obj_pcona.read_parameters()
        obj_pcona.calc_Rdash()

        obj_pcona.calc_contact_angle_with_averaging()
        obj_pcona.plot_contact_angle()
        obj_pcona.calc_contact_angle_between_inlet_outlet(self.t_entry, self.t_exit)
        self.ca_angle = obj_pcona.theta
        self.x_1 = obj_pcona.x_1
        self.x_2 = obj_pcona.x_2
        print(self.x_1, self.x_2)
        print("DONE: You can close the window")
        logging.info('SUCCESS!')

    def onStop(self, event):

        """
        Terminates the script
        """
        self.Destroy()


if __name__ == '__main__':

    app = wx.App()
    TimingChannelGUI(None, title='Channel Analysis')

    app.MainLoop()
