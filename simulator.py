# /usr/local/bin/python3
# 
# - provides framework to read, visualize, and simulate bathymetric data

import math, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from pathlib import Path
from utils import Animation, BoundingBox, FileFormatError


class Bathymetry():
    """
    bathymetry object used for parsing, loading, plotting, and simulating data
        + multibeam bathymetry is treated as ground truth
        + scanning sonar readings are treated as observations
    """
    def __init__(self, filename, bb, animations):
        self.filename = filename
        self.bb = bb
        self.animations = animations
        self.header_size = 6
        

    @staticmethod
    def parse_file(filename, bb, animations=True, save=True):
        """
        creates new Bathymetry object by parsing file 
        :param filename: the location of the bathymetry file to be parsed
            + requires that filename is a valid .asc file
            + requires that the file is in proper ASCII grid format
        :param save: indicates whether or not the parsed data is saved
        :param bb: the bounding box of the plot of bathymetry
            + bounding box in units of degrees longitude and latitude
        :raises FileNotFoundError: if filepath does not exist
        :raises FileFormatError: if file does not follow ASCII Grid format 
        :returns: the parsed bathymetry object
        """ 
        if animations: print('>> input file: ' + filename)
        bath = Bathymetry(filename, bb, animations)
        bath._parse_file(save)
        return bath


    @staticmethod
    def load_file(filename, bb, animations=True):
        """
        loads new Bathymetry object from a file that has already been parsed
        :param filename: location of the original bathymetry file 
            + requires that the bathymetry file has already been parsed
              and that the parsed file is located in the same directory
              with the same name (but with different file extension)
        :param bb: the bounding box of the plot of bathymetry
            + bounding box in units of degrees longitude and latitude
        :raises FileNotFoundError: if file has not been parsed properly
        """
        if animations: print('>> input file: ' + filename)
        # filename is required have .asc extension (hence the value -4)
        ext_len = -4
        bath_header_file =    filename[:ext_len] + '_header.npy'
        bath_multibeam_file = filename[:ext_len] + '_multibeam.npy'

        # raise exception if parsed header and multibeam files are not found
        if (not(Path(bath_header_file).is_file()) or 
            not(Path(bath_multibeam_file).is_file())):
            raise FileNotFoundError('must parse file before loading file')
        
        # otherwise, load the file
        else:
            if animations: print('>> loading file')

            # initialize a new Bathymetry object
            bath = Bathymetry(filename, bb, animations)

            # load the header and multibeam files into new object
            bath_header =    np.load(bath_header_file).item()
            bath_multibeam = np.load(bath_multibeam_file)
            bath._set_header(bath_header)
            bath._set_multibeam(bath_multibeam)
            return bath


    def _parse_file(self, save):
        """
        parses file by parsing header and then body of the file
        """
        # check if filepath exists
        if Path(self.filename).is_file():

            # parse the header and body of the same file
            file = self._parse_header()
            self._parse_body(file)

            # save the header and multibeam data in same location as filename
            if save:
                # filename is required have .asc extension (hence the value -4)
                ext_len = -4
                bath_header_file =    self.filename[:ext_len] + '_header.npy'
                bath_multibeam_file = self.filename[:ext_len] + '_multibeam.npy'

                # save the header and multibeam files
                if self.animations: print('>> saving file')
                np.save(bath_header_file,    self.header)
                np.save(bath_multibeam_file, self.multibeam)

        # file cannot be located in the directory given
        else: 
            raise FileNotFoundError('file not found, check directory')


    def _parse_header(self):
        """
        parse the header of the file
        :returns: the file object so that the body can be parsed
        """
        # initialize header dictionary
        header = {}
        file = open(self.filename, 'r')

        # valid fields that could be found in the header
        valid_fields = set(['ncols', 'nrows', 'xllcorner', 'xllcenter',
                            'yllcorner', 'yllcenter', 'cellsize', 
                            'nodata_value'])
        # fields that are maintained as part of the header
        header_fields = set(['ncols', 'nrows', 'cellsize', 'nodata_value'])  

        # iterate over the header elements
        for i in range(self.header_size):
            header_line = [s.lower() for s in file.readline().split()]

            # check to make sure the header is valid ASCII Grid format
            if ((len(header_line) != 2) or
                (header_line[0] not in valid_fields)):
               raise FileFormatError('invalid header')

            # handle each header field
            elif header_line[0] in header_fields: 
                header[header_line[0]] = self._parse_number(header_line[1])

        self._set_header(header)
        return file


    def _parse_body(self, file):
        """
        parses the body of the file
        :param file: the file that has already had its header parsed
        """ 
        # initialize array to maintain the multibeam bathymetry data 
        self.multibeam = np.empty((self.nrows, self.ncols))
        end_file = ''
        body_i = 0
        body_line = file.readline()

        # create animation to update user of parsing progress
        if self.animations: 
            parse_animation = Animation(state='parsing file', size=self.nrows)

        # check to make sure that all rows contain correct number of elements
        while body_line != end_file:
            # check that number of columns is correct
            if len(body_line.split()) != self.ncols:
                raise FileFormatError('invalid number of cols')

            self.multibeam[body_i] = np.array([self._parse_number(s) for s in body_line.split()])
            body_i += 1
            body_line = file.readline()

            # update the animation to indicate process status
            if self.animations: 
                parse_animation.animate(body_i)

        # check that number of rows is correct
        if body_i != self.nrows:
            raise FileFormatError('invalid number of rows')


    def _parse_number(self, str_number):
        """
        parses a string into either an int or a float as necessary
        :param str_num: a string that represents a numerical value
        :returns: the value of the string, either int or float
        """
        default_nodata_value = -9999
        # try casting as float to assess numerical strings
        try: 
            float_value = float(str_number)
            if '.' in str_number: return float_value
            else:                 return int(float_value)

        # otherwise return default value
        except ValueError:
            return default_nodata_value 


    def _set_header(self, header):
        """
        updates the header field
        :param header: the header to be set
        """
        self.header = header
        self.nrows = header['nrows']
        self.ncols = header['ncols']
        self.resolution = header['cellsize']
        self.nodata_value = header['nodata_value']


    def _set_multibeam(self, multibeam):
        """
        updates the multibeam field
        :param multibeam: the multibeam field to be set
        """
        self.multibeam = multibeam


    def plot(self, title='Bathymetry Data', bb=None, resolution='default'):
        """
        extracts and plots a rectangle of bathymetry
        :param bb: the bounding box of bathymetry to be plotted
            + bounding box in units of degrees longitude and latitude
            + requires that the input bounding box is a subset of the data
            + default behavior is to plot all available data (no bb given)
        """
        if self.animations: print('>> plotting bathymetry')

        # array skipping for when a rought plot is desired
        if   resolution == 'high'         : skip =   1 
        elif resolution == 'intermediate' : skip =   5
        if   resolution == 'crude'        : skip =  50
        else                              : skip =  10

        # get indices for bathymetry to be plotted
        if bb != None:
            # get intersection between the data set and input bounding boxes
            bb_int = self.bb.get_intersection(bb)
            row_min = math.floor(((self.bb.get_n_lim() - bb_int.get_n_lim()) / self.bb.get_lat_range()) * self.nrows)
            row_max = math.floor(((self.bb.get_n_lim() - bb_int.get_s_lim()) / self.bb.get_lat_range()) * self.nrows)
            col_min = math.floor(((bb_int.get_w_lim() - self.bb.get_w_lim()) / self.bb.get_lon_range()) * self.ncols)
            col_max = math.floor(((bb_int.get_e_lim() - self.bb.get_w_lim()) / self.bb.get_lon_range()) * self.ncols)
        
        # default behavior (bb not given) is to plot all of the data
        else:
            row_min, col_min = 0,0
            row_max, col_max = self.multibeam.shape

        # extract the corresponding rectangle of multibeam data
        bath_rect = self.multibeam[row_min:row_max:skip, col_min:col_max:skip]
        bath_set = set(bath_rect.flatten())

        # remove nodata_value to achieve appropriate color scaling
        if self.nodata_value in bath_set:
            bath_set.remove(self.nodata_value)

        # calculate the max in min bathymetry height to set the color scale
        vmin = min(bath_set)
        vmax = max(bath_set)

        # mask to ignore 'nodata_value' when plotting
        def mask(x):
            return x == self.nodata_value

        # plotting parameters
        figsize = (10, 6.5)
        font_large = 22
        font_medium = 15
        font_small = 12
        line_width = 5
        x_tick_num = 5
        y_tick_num = 5

        # setting label tick spacing
        rows = bath_rect.shape[0]
        cols = bath_rect.shape[1]
        x_tick_spacing = math.ceil(cols/x_tick_num)
        y_tick_spacing = math.ceil(rows/y_tick_num)

        # plot the figure
        fig = plt.figure(figsize=figsize)
        sns.set_style('darkgrid')
        ax = sns.heatmap(bath_rect, square=True, cmap='jet', 
                         vmin=vmin, vmax=vmax, mask=mask(bath_rect),
                         xticklabels=True, yticklabels=True, 
                         cbar_kws={'label': 'Depth [Meters]'})
        ax.figure.axes[-1].yaxis.label.set_size(font_medium)
        ax.figure.axes[-1].set_yticklabels(ax.figure.axes[-1].get_yticklabels(), size=font_small)

        # set title, axis labels, and tick labels
        ax.set_title(title, fontsize=font_large)
        ax.set_xlabel('Longitude [degrees]', fontsize=font_medium)
        ax.set_ylabel('Latitude [degrees]', fontsize=font_medium)
        ax.set_xticks(ax.get_xticks()[int(x_tick_spacing/2)::x_tick_spacing])
        ax.set_yticks(ax.get_yticks()[int(y_tick_spacing/2)::y_tick_spacing])
        ax.set_xticklabels([str(round(self.bb.get_lon(a/cols), 2)) for a in ax.get_xticks()], rotation=0, fontsize=font_small)
        ax.set_yticklabels([str(round(self.bb.get_lat(a/rows), 2)) for a in ax.get_yticks()], fontsize=font_small)
        fig.savefig('bathymetry/plots/low.png')



if __name__ == '__main__':
    # raw data set
    #   + download link: (http://www.soest.hawaii.edu/pibhmc/cms/)
    raw_file  = 'bathymetry/kohala/kohala_synth_5m.asc'
    raw_bb = BoundingBox(w_lim = -156.31, 
                         e_lim = -155.67, 
                         n_lim =   20.54, 
                         s_lim =   19.64)

    # Falkor data set where engineering cruise took place
    #   + more information about Falkor: (https://schmidtocean.org/rv-falkor/)
    falkor_file = 'bathymetry/falkor/falkor_5m.npy'
    falkor_bb = BoundingBox(w_lim = -156.03, 
                            e_lim = -155.82, 
                            n_lim =   20.01, 
                            s_lim =   19.84)

    falkor_bath = Bathymetry.load_file(falkor_file, falkor_bb)
    falkor_bath.plot(title='Hawaii, HI', resolution='crude')

