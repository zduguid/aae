# /usr/local/bin/python3
# 
# - utilities for reading, visualizing, and simulating bathymetric data

import math
import sys

class Animation():
    def __init__(self, state, size):
        """
        initialize Animation object
            + used to update user on the status of a process
            + helpful for working with large data sets
        :param state: string of text that describes the program state
            + example = "parsing file"
        :param size: the amount of iterations until the process is complete
            + requires that size corresponds to the process loop size
        """
        self.state = state
        self.size = size
        self.percent = 0
        self.text_index = 0
        self.text_len = 8
        self.running = False
        self.line = ">> " + self.state + ": "
        self.percent_text = "(" + str(self.percent) + "%)"
        self.text = ["[" + " "*(i) + "=" + " "*(self.text_len-3-i) + "]" 
                     for i in range(self.text_len-2)]

    def animate(self, i):
        """
        updates the animation 
            + requires that animate function is called inside the
              execution loop of some process 
        :param i: describes the ith iteration of executing the process
        """
        base = 100

        # write first frame of animation
        if not(self.running):
            sys.stdout.write(self.line + 
                             self.text[self.text_index] + 
                             self.percent_text)
            sys.stdout.flush()
            self.running = True

        # update frame of animation
        elif (math.floor((i+1)/self.size*base) != self.percent):
            # erase previous frame of animation
            sys.stdout.write('\b'*(self.text_len + len(self.percent_text)))

            # update animation parameters
            self.percent = math.floor((i+1)/self.size*base)
            self.percent_text = '(' + str(self.percent) + '%)'
            self.text_index = (self.text_index + 1) % len(self.text)

            # write next frame of animation
            if self.percent != base:
                sys.stdout.write(self.text[self.text_index] + self.percent_text)
                sys.stdout.flush()
            else:
                sys.stdout.write('['+'='*(self.text_len-2)+']' + self.percent_text + '\n')
                sys.stdout.flush()
                

class BoundingBox():
    def __init__(self, w_lim, e_lim, n_lim, s_lim):
        """
        bounding box that corresponding to a patch of bathymetry
        :param w_lim: the western  limit of the bounding box (longitude)
        :param e_lim: the eastern  limit of the bounding box (longitude)
        :param n_lim: the northern limit of the bounding box (latitude)
        :param s_lim: the southern limit of the bounding box (latitude)
        """
        self.w_lim = w_lim
        self.e_lim = e_lim
        self.n_lim = n_lim
        self.s_lim = s_lim
        self.lon_range = self.e_lim - self.w_lim
        self.lat_range = self.n_lim - self.s_lim
        assert self.lon_range > 0
        assert self.lat_range > 0

    def get_w_lim(self):
        return self.w_lim

    def get_e_lim(self):
        return self.e_lim

    def get_n_lim(self):
        return self.n_lim

    def get_s_lim(self):
        return self.s_lim

    def get_lon_range(self):
        return self.lon_range

    def get_lat_range(self):
        return self.lat_range

    def get_lon(self, ratio):
        """
        calculates the correct longitude for a given location
        :param ratio: the ratio of longitude between western and eastern limits
        :returns: the longitude value equal to the ratio of the bounding box
        """
        return (ratio * self.get_lon_range()) + self.get_w_lim()

    def get_lat(self, ratio):
        """
        calculates the correct latitude for a given location
        :param ratio: the ratio of latitude between southern and northern limits
        :returns: the longitude value equal to the ratio of the bounding box
        """
        return -(ratio * self.get_lat_range()) + self.get_n_lim()

    def get_intersection(self, bb):
        """
        determines the intersection of two bounding boxes, used for plotting
        :param bb: the input bounding box to compute the intersection with self
        :returns: a new bounding box, representing the intersection 
        """
        w_lim = max(self.get_w_lim(), bb.get_w_lim())
        e_lim = min(self.get_e_lim(), bb.get_e_lim())
        n_lim = min(self.get_n_lim(), bb.get_n_lim())
        s_lim = max(self.get_s_lim(), bb.get_s_lim())
        return BoundingBox(w_lim, e_lim, n_lim, s_lim)


class FileFormatError(Exception):
    """
    Exception to raise when data contained by file is in incorrect format
    """
    pass