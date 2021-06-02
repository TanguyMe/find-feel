#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""visualisation_tool.py: Regroups function to help in projects"""

__author__ = "Antoine"
__copyright__ = "Copyright 2021, The Project"
__credits__ = ["Antoine"]
__version__ = "1.0"
__status__ = "Production"

import os
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import barplot as barplot
from seaborn import regplot as regplot
from IPython.display import Image



def make_plot(x, y,  type_plot, name, figsize_lon=25, figsize_lar=6, data=None, axis_labels=None, override=False, legende=None, xlabel=None,
ylabel=None ):
    """
        Save the pairplots as an image without rendering it ,
        then it display the image.

        params:
            x(series): x axis datas,
            y(series): y axis datas,
            type_plot: 
            figsize_lon:
            figsize_lar:
            name(string): name of the file for saving or displaying,
            data :
            override(bool): if True, override saved file
    """
    folder = '../data/visualisation_cache/'
    files = os.listdir(folder)
    filename = "plot_{}.jpg".format(name)
    path = "./{}{}".format(folder, filename)

    if override or filename not in files:
        if  type_plot == barplot:
            print("Saving...")
            plt.figure(figsize=(figsize_lon,figsize_lar))
            if name != None :
                plt.title(name)
            if legende != None :
                plt.legend(legende)
            if axis_labels != None :
                plt.set(axis_labels)
            barplot(x=x,y=y)
            if xlabel != None :
                plt.xlabel(xlabel)
            if ylabel != None :
                plt.ylabel(ylabel)
            plt.savefig(path)
            print("Saved as {} in folder {}".format(filename, path))
            plt.clf()

        if  type_plot == regplot:
            print("Saving...")
            plt.figure(figsize=(figsize_lon,figsize_lar))
            if name != None :
                plt.title(name)
            if legende != None :
                plt.legend(legende)
            if axis_labels != None :
                plt.set(axis_labels)
            regplot(x=x,y=y,data=data)
            if xlabel != None :
                plt.xlabel(xlabel)
            if ylabel != None :
                plt.ylabel(ylabel)
            plt.savefig(path)
            print("Saved as {} in folder {}".format(filename, path))
            plt.clf()
    else:
        print("File already exist")
    display(Image(filename=path))