# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

__description__ = 'Semi-Automatic tool to annotate cells based on their frame-to-frame pixel-shift.'


import os
import wx
import glob
import cv2
import numpy as np

# Preserve the aspect ratio with which the pixel shifts were computed

wildcard = "Python source (*.py)|*.py|" \
            "All files (*.*)|*.*"

class AnnotationTool(wx.Frame):

    def __init__(self, parent, title):
        super(AnnotationTool, self).__init__(parent, title=title)

        self.InitUI()
        self.Centre()

        self.SetSize((640, 640))
        self.currentPicture = 0
        self.totalPictures = 0
        self.posi = []
        self.cell_coord = []

        self.Show()

    def InitUI(self):

        panel = wx.Panel(self, style=wx.BORDER_RAISED)

        self.currentDirectory = os.getcwd()
        sizer = wx.GridBagSizer(3, 3)

        text1 = wx.StaticText(panel, label="Image Folder")
        sizer.Add(text1, pos=(1, 0), flag=wx.LEFT, border=10)
        self.tc1 = wx.TextCtrl(panel, 1)
        sizer.Add(self.tc1, pos=(1, 1), span=(1, 2), flag=wx.TOP|wx.EXPAND)

        line1 = wx.StaticLine(panel)
        sizer.Add(line1, pos=(3, 0), span=(1, 20), flag=wx.EXPAND|wx.BOTTOM, border=10)

        text2 = wx.StaticText(panel, label="Skip Factor")
        sizer.Add(text2, pos=(2, 0), flag=wx.LEFT, border=10)

        self.tc2 = wx.TextCtrl(panel)
        wx.EVT_TEXT(self, self.tc2.GetId(), self.onSkip)
        sizer.Add(self.tc2, pos=(2, 1), span=(1, 1), flag=wx.TOP)

        self.button2 = wx.Button(panel, 2, label="Browse")
        self.button2.SetFocus()
        sizer.Add(self.button2, pos=(1, 3), flag=wx.LEFT, border=10)
        wx.EVT_BUTTON(self, self.button2.GetId(), self.onBrowse)

        self.button3 = wx.Button(panel, 3, label="Load Image")
        self.button3.SetFocus()
        sizer.Add(self.button3, pos=(4, 0), flag=wx.LEFT, border=10)
        wx.EVT_BUTTON(self, self.button3.GetId(), self.onLoad)

        self.button6 = wx.Button(panel, 6, label=">")
        self.button6.SetFocus()
        sizer.Add(self.button6, pos=(6, 0), flag=wx.LEFT, border=10)
        wx.EVT_BUTTON(self, self.button6.GetId(), self.onNext)

        self.button8 = wx.Button(panel, 8, label="Update")
        self.button8.SetFocus()
        sizer.Add(self.button8, pos=(6, 1), flag=wx.LEFT, border=10)
        wx.EVT_BUTTON(self, self.button8.GetId(), self.onUpdate)

        self.button9 = wx.Button(panel, 9, label="Undo")
        self.button9.SetFocus()
        sizer.Add(self.button9, pos=(4, 1), flag=wx.LEFT, border=10)
        wx.EVT_BUTTON(self, self.button9.GetId(), self.onUndo)

        self.label0 = wx.StaticText(panel, 1, label = 'Cell count now: ')
        sizer.Add(self.label0, pos=(8, 0), flag=wx.LEFT, border=10)

        self.label = wx.StaticText(panel, 1, label = '00000')
        self.label.SetBackgroundColour(wx.BLACK)
        self.label.SetForegroundColour(wx.YELLOW)
        font = wx.Font(14, wx.DECORATIVE, wx.ITALIC, wx.NORMAL)
        self.label.SetFont(font)
        sizer.Add(self.label, pos=(8, 1), flag=wx.LEFT, border=10)


        img = wx.EmptyImage(640, 640)
        self.imageCtrl = wx.StaticBitmap(panel, wx.ID_ANY, wx.BitmapFromImage(img))
        self.imageCtrl.Bind(wx.EVT_LEFT_DOWN, self.onView)
        sizer.Add(self.imageCtrl, pos=(5, 0), span=(1, 2), flag=wx.LEFT, border=10)

        panel.SetSizer(sizer)

    def onSkip(self, event):

        self.skip_val = event.GetString()
        self.skip_val = int(self.skip_val)
        return self.skip_val

    def onBrowse(self, event):
        dlg = wx.FileDialog(
                            self, message="Choose a file",
                            defaultDir=self.currentDirectory,
                            defaultFile="",
                            wildcard=wildcard,
                            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
                            )
        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
            print("You chose the following file(s):")
            for path in paths:
                self.path = path


                filepath = os.path.dirname(self.path).replace('\\', '/')
                self.filepath = filepath + ('/')
                self.tc1.SetValue(self.filepath)

                self.tiffFiles = sorted(glob.glob1(self.filepath, "*.tiff"))

                self.filename = self.tiffFiles

                print(self.filename)

        dlg.Destroy()
        os.chdir(self.filepath)
        os.mkdir('Temp')
        self.temp = os.path.join(self.filepath + 'Temp' + '/')

    def onLoad(self, event):

        filepath = self.filename[0]
        img = wx.Image(filepath, wx.BITMAP_TYPE_ANY)

        self.imageCtrl.SetBitmap(wx.BitmapFromImage(img))
        self.Refresh()

    def loadImage(self):

        curr_image = self.filepath + self.filename[self.currentPicture]
        img = wx.Image(curr_image, wx.BITMAP_TYPE_ANY)

        self.imageCtrl.SetBitmap(wx.BitmapFromImage(img))

    def nextPicture(self):

        if self.currentPicture == self.totalPictures-self.skip_val:
            self.currentPicture = 0
        else:
            self.currentPicture += self.skip_val
        self.loadImage()

    def onNext(self, event):

        self.nextPicture()

    def onView(self, event):

        pos = event.GetPosition()

        dc = wx.ClientDC(self.imageCtrl)
        dc.DrawCircle(pos[0], pos[1], 2)
        self.posi.append((pos[0], pos[1]))

    def onUndo(self, event):

        undo_coords = self.posi
        self.posi = []
        undo_coords = undo_coords[:-1]

        temp_val = len(undo_coords)

        self.loadImage()
        dc = wx.ClientDC(self.imageCtrl)

        for i in range(0, temp_val, 1):
            dc.DrawCircle(undo_coords[i][0], undo_coords[i][1], 2)
            self.posi.append((undo_coords[i][0], undo_coords[i][1]))

        print(self.posi)

    def saveImage(self):

        self.fname = os.path.splitext(self.filename[self.currentPicture])[0][:5]
        np.savetxt(self.temp + str(self.fname) + '.csv', self.posi, delimiter=",", fmt='%.9f')
        self.posi = []

        coords = np.genfromtxt(self.temp + str(self.fname) + '.csv', dtype =int, delimiter = ",")

        x = coords[:, 0]

        y = coords[:, 1]
        val = len(coords)

        im1 = cv2.imread(self.filepath + self.filename[self.currentPicture], 0)
        print(self.filepath + self.filename[self.currentPicture])

        input_image =cv2.cvtColor(im1,cv2.COLOR_GRAY2BGR)

        for i in range(0, val, 1):
            cv2.circle(input_image,(x[i], y[i]),1,(0,0,255),1)

        cv2.imwrite(self.temp + str(self.fname) + '.png', input_image)

    def pixelShift(self):

        pixel_coords = np.genfromtxt(self.temp + str(self.fname) + '.csv', dtype =int, delimiter = ",")

        x = pixel_coords[:, 0]
        y = pixel_coords[:, 1]

        val = len(pixel_coords)

        self.label.SetLabel(format(val))

        ps_image = self.filepath + self.filename[self.currentPicture]
        img = wx.Image(ps_image, wx.BITMAP_TYPE_ANY)

        self.imageCtrl.SetBitmap(wx.BitmapFromImage(img))
        self.readPixelShift()
        dc = wx.ClientDC(self.imageCtrl)
        for i in range(0, val, 1):
            dc.DrawCircle(x[i], y[i]+self.yshift, 2)
            self.posi.append((x[i], y[i]+self.yshift))

    def readPixelShift(self):

        shift = []
        pixel_shifts = np.genfromtxt(self.filepath + 'pixel_shifts.csv', dtype =float, delimiter = ",")
        px = pixel_shifts[:, 1].tolist()
        index_val = px.index(int(self.fname))

        skip_index_val = px.index(int(self.fname)+self.skip_val)

        for j in range(index_val, skip_index_val, 1):
            tmp = pixel_shifts[:, 0][j]
            shift += [tmp]

        self.yshift = np.sum(shift)

        self.yshift = int(round(self.yshift))
        print(self.yshift)


    def onUpdate(self, event):

        self.saveImage()
        self.onNext(event)
        self.pixelShift()


if __name__== '__main__':

    app = wx.App()
    AnnotationTool(None,  title='PrecisionTest: Annotation Tool')
    app.MainLoop()
