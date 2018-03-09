import cv2
import math
import numpy
#import IPython
import Dubins
import Utils
import KinematicModel as model

class ObstacleManager(object):

  def __init__(self, mapMsg):
    # Setup the map
    self.map_info = mapMsg.info
    self.mapImageGS = numpy.array(mapMsg.data, dtype=numpy.uint8).reshape((mapMsg.info.height, mapMsg.info.width,1))

    # Retrieve the map dimensions
    height, width, channels = self.mapImageGS.shape
    self.mapHeight = height
    self.mapWidth = width
    self.mapChannels = channels

    # Binarize the Image
    self.mapImageBW = 255*numpy.ones_like(self.mapImageGS, dtype=numpy.uint8)
    self.mapImageBW[self.mapImageGS==0] = 0 # Hiro's NOTE: 0 means valid, 1 means invalid
    self.mapImageBW = self.mapImageBW[::-1,:,:] # Need to flip across the y-axis

    # Obtain the car length and width in pixels
    self.robotWidth = int(model.CAR_WIDTH/self.map_info.resolution + 0.5)
    self.robotLength = int(model.CAR_LENGTH/self.map_info.resolution + 0.5)

  # Check if the passed config is in collision
  # config: The configuration to check (in meters and radians)
  # Returns False if in collision, True if not in collision
  def get_state_validity(self, config):

    # Convert the configuration to map-coordinates -> mapConfig is in pixel-space
    mapConfig = Utils.world_to_map(config, self.map_info)

    # ---------------------------------------------------------
    # YOUR CODE HERE
    #
    # Return true or false based on whether the configuration is in collision
    # Use self.robotWidth and self.robotLength to represent the size of the robot
    # Also return false if the robot is out of bounds of the map
    # Although our configuration includes rotation, assume that the
    # rectangle representing the robot is always aligned with the coordinate axes of the
    # map for simplicity
    # ----------------------------------------------------------


    # This is the section where I wrote the code
    # CONFIG IS JUST POSE
    # For simplicity I'm going to just set square aligned to the coordinate axis
    # as recommended

    print "We're in get_state_validity"
    half_width = int(self.robotWidth/2)
    half_length = int(self.robotLength/2)

    x = mapConfig[0]
    y = mapConfig[1]
    angle = mapConfig[2]

    for i in range(-half_width, half_width):
        for j in range(-half_length, half_length):
            if(self.mapImageBW[x + i, y+j] == 1):
                print "We're exiting get_state_validity"
                return False  # Hrio's NOTE: Meaning that is's invalid
    print "We're exiting get_state_validity"

    # This is where the section end

    return True     # Hiro's NOTE: Meaning that it's valid

  # Check if there is an unobstructed edge between the passed configs
  # config1, config2: The configurations to check (in meters and radians)
  # Returns false if obstructed edge, True otherwise
  def get_edge_validity(self, config1, config2):
    # -----------------------------------------------------------
    # YOUR CODE HERE
    #
    # Check if endpoints are obstructed, if either is, return false
    # Find path between two configs using Dubins
    # Check if all configurations along Dubins path are obstructed
    # -----------------------------------------------------------

    # CONFIG IS JUST POSE
    print "We're entering get_edge_validitiy"
    print "Checking if endpoint is obstructed"

    if  not get_state_validity(config1) or not get_state_validity(config2):
        print "Endpoint is obstructed"
        return False
    print "Endpoint isn't obstructed"

    curvature = .5  # Some random value for now

    # Hiro's NOTE: What is the curvature?
    ppx, ppy, ppyaw, pclen = dubins_path_planning(config1, config2, curvature)

    print "Checking if path is obstructed:"

    for  i in range(len(ppx)):
        config = [ppx[i], ppy[i], ppyaw[i]]
        if not get_state_validity(config):
            print "Path is obstructed"
            return False
    print "Path is not obstructed"


    print "We're exiting get_edge_validity

    # This is where the section end



    return True


# Test
if __name__ == '__main__':
  # Write test code here!
  pass
