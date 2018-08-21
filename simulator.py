# /usr/local/bin/python3
# 
# - provides framework to read, visualize, and simulate bathymetric data

import math, sys, random
import numpy as np
import matplotlib.pyplot as plt
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
    def parse_file(filename, bb, save=True, animations=True):
        """
        creates new Bathymetry object by parsing file 
        :param filename: the location of the bathymetry file to be parsed
            + requires that filename is a valid .asc file
            + requires that the file is in proper ASCII grid format
        :param save: indicates whether or not the parsed data is saved
        :param bb: the bounding box of the plot of bathymetry
            + bounding box in units of degrees longitude and latitude
        :param animations: indicates whether command-line updates are displayed
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
        :param animations: indicates whether command-line updates are displayed
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
        :param save: indicates whether or not the parsed data is saved
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
                bath_header_file =    self.filename[:ext_len]+'_header.npy'
                bath_multibeam_file = self.filename[:ext_len]+'_multibeam.npy'

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

            # store each new row of data in the array
            self.multibeam[body_i] = np.array([self._parse_number(s) 
                                               for s in body_line.split()])
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
        self.header       = header
        self.nrows        = header['nrows']
        self.ncols        = header['ncols']
        self.resolution   = header['cellsize']
        self.nodata_value = header['nodata_value']


    def _set_multibeam(self, multibeam):
        """
        updates the multibeam field
        :param multibeam: the multibeam field to be set
        """
        self.multibeam = multibeam


    def plot_bathymetry(self, title='Bathymetry Data', bb=None, resolution=''):
        """
        extracts and plots a rectangle of bathymetry
        :param title: the title of the created plot
        :param bb: the bounding box of bathymetry to be plotted
            + bounding box in units of degrees longitude and latitude
            + requires that the input bounding box is a subset of the data
            + default behavior is to plot all available data (no bb given)
        :param resolution: controls resolution of the graph output 
            + lower resolution ('crude' option) is plotted much faster
        """
        if self.animations: print('>> plotting bathymetry')

        # array skipping for when a rought plot is desired
        if   resolution == 'high'         : skip =   1 
        elif resolution == 'intermediate' : skip =   5
        elif resolution == 'crude'        : skip =  50
        else                              : skip =  10

        # get indices for bathymetry to be plotted
        if bb != None:
            # get intersection between the data set and input bounding boxes
            bb_int = self.bb.get_intersection(bb)
            row_min = math.floor(((self.bb.get_n_lim() - bb_int.get_n_lim())  
                                 / self.bb.get_lat_range()) * self.nrows)
            row_max = math.floor(((self.bb.get_n_lim() - bb_int.get_s_lim())  
                                 / self.bb.get_lat_range()) * self.nrows)
            col_min = math.floor(((bb_int.get_w_lim()  - self.bb.get_w_lim())
                                 / self.bb.get_lon_range()) * self.ncols)
            col_max = math.floor(((bb_int.get_e_lim()  - self.bb.get_w_lim())
                                 / self.bb.get_lon_range()) * self.ncols)
        
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
        ax.figure.axes[-1].set_yticklabels(ax.figure.axes[-1].get_yticklabels(), 
                                           size=font_small)

        # set title, axis labels, and tick labels
        ax.set_title(title, fontsize=font_large)
        ax.set_xlabel('Longitude [degrees]', fontsize=font_medium)
        ax.set_ylabel('Latitude [degrees]', fontsize=font_medium)
        ax.set_xticks(ax.get_xticks()[int(x_tick_spacing/2)::x_tick_spacing])
        ax.set_yticks(ax.get_yticks()[int(y_tick_spacing/2)::y_tick_spacing])
        ax.set_xticklabels([str(round(self.bb.get_lon(a/cols), 2))
                            for a in ax.get_xticks()], fontsize=font_small)
        ax.set_yticklabels([str(round(self.bb.get_lat(a/rows), 2))
                            for a in ax.get_yticks()], fontsize=font_small)
        fig.savefig('data/plots/5m_' + resolution + '.png')
        plt.close()


    def simulate_sonar_data(self, n, patch_length=50, plot=False):
        """
        simulates example sonar data readings to mimic AUG scanning sonar
        :param n: number of data points to be produced
        :param patch_length: size of matrix to represent 5m-gridded bathymetry
        :param plot: indicates whether or not simulated data is plotted
        :returns: a tuple of arrays where
            + the first  array is n patches of ground truth bathymetry
            + the second array is n patches of simulated sonar readings
        """
        # initialize data array
        if self.animations: print('>> simulating data')

        # data arrays used to maintain valid samples
        data_bath  = []
        data_sonar = []

        # depth allowed allowed in patch (enforces operational feasibility)
        depth_threshold_low  = -150
        depth_threshold_high =   20
        patch_buffer = int(patch_length/2)

        # tolerance allows for small amount of missing data (1% of patch area)
        nodata_tolerance = int(0.01 * patch_length**2)

        # sample row and col are maintained to make location-based plots
        sample_rows = []
        sample_cols = []

        # helper function to simulate a sample and update plotting parameters
        def add_new_sample(patch_bath):
            point_bath, point_sonar = self._simulate(patch_bath)
            data_bath.append(point_bath)
            data_sonar.append(point_sonar)
            sample_rows.append(-row)
            sample_cols.append(col)

        # perform random sampling until specified amount of data acquired
        while len(data_bath) < n:

            # randomly sample a patch of bathymetry
            col = np.random.randint(patch_buffer, 
                                    self.multibeam.shape[1] - patch_buffer)
            row = np.random.randint(patch_buffer, 
                                    self.multibeam.shape[0] - patch_buffer)
            patch_bath = self.multibeam[row-patch_buffer:row+patch_buffer, 
                                        col-patch_buffer:col+patch_buffer]

            # filter out samples that are too shallow for operations
            if np.max(patch_bath) < depth_threshold_high:

                # when samples not too deep they are valid samples
                if np.min(patch_bath) > depth_threshold_low:
                    add_new_sample(patch_bath) 

                # otherwise count the frequency of the nodata_value
                else:

                    # get next lowest value if frequency is low enough
                    unique, counts = np.unique(patch_bath, return_counts=True)
                    if (unique[0] == self.nodata_value and 
                        counts[0] < nodata_tolerance):
                        patch_set = set(patch_bath.flatten())
                        patch_set.remove(self.nodata_value)

                        # when samples not too deep they are valid samples
                        if min(patch_set) > depth_threshold_low:

                            # set nodata to the median for stable learning
                            median = np.median(patch_bath)
                            patch_bath[patch_bath==self.nodata_value] = median
                            add_new_sample(patch_bath) 

        # convert the list to a numpy array
        data_bath  = np.array(data_bath)
        data_sonar = np.array(data_sonar)

        if plot == True:
            # plot locations of where data was simulated
            self._plot_sample_locations(sample_cols, sample_rows, 
                                        self.multibeam.shape)

            # plot subset of data that was simulated
            self._plot_simulated_data(data_bath, data_sonar)

        return data_bath, data_sonar


    def _simulate(self, patch_bath):
        """
        simulates sonar readings to mimic that of real AUG sonar measurements
        :param patch: the bathymetry patch that the AUG is measuring
        :returns: two matrices both with shape (path_length, patch_length)
            + where the 1st matrix represents the sample bathymetry patch
            + where the 2nd matrix represents the simulated sonar readings
                + such that both matrices have been scaled to the range [-1,1]
        """
        # defensive copying of the input patch
        patch_bath = np.copy(patch_bath)
        # the matrix dimensions of the bathymetry patch
        patch_length = patch_bath.shape[0]
        # ascent/descent pitch angle in [rad]
        pitch_angle = 0.45
        # depth band in [m]
        depth_band = (patch_length/1.5) / math.tan(pitch_angle)
        # resolution of grid in [m]
        grid = 5 
        # horizontal glider speed in [m/s]
        horizontal_speed = 1
        # dive rate of glider in [m/s]
        dive_rate = horizontal_speed * math.tan(pitch_angle)
        # time to complete one ascent-descent cycle in [s]
        T = patch_length / horizontal_speed
        # randomly selected heading in [rad]
        theta0 = np.random.uniform(0, 2*np.pi)

        # other constants used when simulating data
        z0          = depth_band * 1.15 / grid
        s0          = 0.1*z0
        s           = s0
        peaks_num   = 10
        peaks_gain  = 2
        local_mu    = 0
        local_sig   = patch_length*0.003
        sonar_mu    = 0
        sonar_sig   = 0.1
        delta_s     = dive_rate
        delta_mu    = 0
        delta_sig   = 0.03 
        theta_mu    = 0
        theta_sig   = 0.4
        d_theta     = abs(np.random.normal(theta_mu, theta_sig))
        bias_window = 1/4
        max_bias    = int(patch_length*bias_window)
        x_bias      = np.random.choice(range(-max_bias, max_bias))
        y_bias      = np.random.choice(range(-max_bias, max_bias))

        # parameterization of glider motion
        t     = np.linspace(-T/2, T/2, peaks_num*patch_length)
        theta = np.linspace(theta0-d_theta/2, theta0+d_theta/2, len(t))
        x_t   = t * horizontal_speed * np.cos(theta)
        y_t   = t * horizontal_speed * np.sin(theta) 
        z_t   = z0 - np.absolute(t)  * dive_rate
        s_t   = np.array([s0])

        # calculate scanning sonar motion in the local frame
        for i in range(len(t)-1):

            # direction of scanning sonar motion changes
            if np.absolute(s + delta_s) >= (np.absolute(z_t[i+1]) + s0):
                delta_s *= -1
                delta_s += np.random.normal(delta_mu, delta_sig)

            # update the state of the scanning sonar in the local frame
            s  += delta_s
            s_t = np.append(s_t, s*peaks_gain)

        # convert sonar motion from local frame to intertial frame
        s_x_t = x_t + s_t * np.sin(theta) + x_bias
        s_y_t = y_t - s_t * np.cos(theta) + y_bias

        # add some random noise to the location of the sonar readings
        s_x_t += np.random.normal(local_mu, local_sig, len(t))
        s_y_t += np.random.normal(local_mu, local_sig, len(t)) 

        # generate index coordinate list of the sonar readings
        coord_list = []
        for i in range(len(t)):
            x = math.floor(s_x_t[i] + patch_length/2)
            y = math.floor(s_y_t[i] + patch_length/2)

            # only include coordinates within the correct range
            if ((x >= 0) and (x <= patch_length - 1) and 
                (y >= 0) and (y <= patch_length - 1)):
                coord_list.append((x,y))

        # extract depth values from the coordinates and add some noise
        patch_sonar_vals  = np.array([patch_bath[j,i] for (i,j) in coord_list])
        patch_sonar_vals += np.random.normal(sonar_mu, sonar_sig, 
                                             len(patch_sonar_vals))
        
        # get the min and max values contained by the patch
        patch_min = np.min(patch_bath)
        patch_max = np.max(patch_bath)

        # perform min-max feature scaling such all values are in range [-1,1]
        patch_sonar_vals -= (patch_max + patch_min)/2
        patch_sonar_vals *= 2/(patch_max - patch_min)
        patch_bath       -= (patch_max + patch_min)/2
        patch_bath       *= 2/(patch_max - patch_min)

        # reconstruct the sonar matrix once features have been scaled
        patch_sonar = np.zeros([patch_length, patch_length])
        for k in range(len(coord_list)):
            i,j = coord_list[k][0], coord_list[k][1]
            patch_sonar[j,i] = patch_sonar_vals[k]

        return patch_bath, patch_sonar


    def _plot_sample_locations(self, sample_col, sample_row, bath_shape):
        """
        plots the longitude-latitude locations of where samples were taken
        :param sample_col: index list of the col where each sample was taken
        :param sample_row: index list of the row where each sample was taken
        """
        # plotting parameters
        font_large = 15
        font_medium = 12
        df = pd.DataFrame()
        df['x'] = sample_col
        df['y'] = sample_row
        x_tick_num = 5
        y_tick_num = 5

        # setting label tick spacing
        rows = bath_shape[0]
        cols = bath_shape[1]
        x_tick_spacing = math.ceil(cols/(x_tick_num+1))
        y_tick_spacing = math.ceil(rows/(y_tick_num+1))
        y_ticks = range(rows)[int(y_tick_spacing/2)::y_tick_spacing][::-1]
        x_ticks = range(cols)[int(x_tick_spacing/2)::x_tick_spacing]
        y_ticks = [str(round(self.bb.get_lat(a/rows), 2)) for a in y_ticks]
        x_ticks = [str(round(self.bb.get_lon(a/cols), 2)) for a in x_ticks]

        # generate plot
        fig = plt.figure()
        sns.set_style('darkgrid')
        g = sns.lmplot('x', 'y', data=df, fit_reg=False, size=6, aspect=1.5,
                       scatter_kws={'s': 1})
        g.set(yticks=range(-rows,0)[int(y_tick_spacing/2)::y_tick_spacing],
              xticks=range(cols)[int(x_tick_spacing/2)::x_tick_spacing],
              yticklabels=y_ticks, xticklabels=x_ticks)
        plt.xlabel('Latitude [degrees]', fontsize=font_medium)
        plt.ylabel('Longitude [degrees]', fontsize=font_medium)
        plt.title('Sample Locations \n (n = ' + str(len(sample_col)) + ')', fontsize=font_large)
        plt.tight_layout()
        plt.savefig('data/plots/simulated_locations.png')
        plt.close()


    def _plot_simulated_data(self, data_bath, data_sonar):
        """
        plots the simulated sonar data with the corresponding bathymetry patch
        :param data: the array of data points containing the simulated data
        """
        if self.animations: print('>> plotting simulated data')

        # plotting parameters
        n = len(data_bath)
        ncols = min(4, n)
        figsize = (14, 6)
        font_large = 25
        font_medium = 15
        sns.set_style('darkgrid')

        # randomly sample points in the data set
        data_indices = random.sample(range(n), ncols)
        data_points_bath  = data_bath[data_indices]
        data_points_sonar = data_sonar[data_indices]

        # mask used for plotting simulated sonar readings
        def mask(x):
            return x == 0

        # initialize the plotting objects
        fig, ax = plt.subplots(figsize=figsize, ncols=ncols, nrows=2)
        plt.subplots_adjust(left    =  0.1,     # left side location
                            bottom  =  0.1,     # bottom side location
                            right   =  0.9,     # right side location
                            top     =  0.9,     # top side location
                            wspace  =  0.6,     # horizontal gap
                            hspace  =  0.05)    # vertical gap 

        # generate each column of the plot
        for i in range(ncols):
            point_bath  = data_points_bath[i]
            point_sonar = data_points_sonar[i]

            # extract min and max for color scaling
            vmin = np.min(point_bath)
            vmax = np.max(point_bath)

            # plot the sonar simulated data
            sns.heatmap(point_bath, square=True, cmap='jet', 
                        vmin=vmin, vmax=vmax, ax=ax[0][i],
                        xticklabels=False, yticklabels=False,
                        mask=mask(point_bath),
                        cbar=False)

            # plot the corresponding patch of bathymetry
            sns.heatmap(point_sonar, square=True, cmap='jet', 
                        vmin=vmin, vmax=vmax, ax=ax[1][i],
                        xticklabels=False, yticklabels=False,
                        mask=mask(point_sonar),
                        cbar=False)

        # handle labeling of subplots
        fig.suptitle('Simulated Sonar Readings', fontsize=font_large)
        ax[0][0].set(ylabel='Sample Bathymetry \n Patch')
        ax[0][0].yaxis.label.set_size(font_medium)
        ax[1][0].set(ylabel='Simulated Sonar \n Readings')
        ax[1][0].yaxis.label.set_size(font_medium)
        for i in range(ncols):
            ax[1][i].set(xlabel='Example ' + str(i+1))
            ax[1][i].xaxis.label.set_size(font_medium)
        plt.savefig('data/plots/simulated_examples.png')
        plt.close()