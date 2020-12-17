from __future__ import print_function
import msgpackrpc #install as admin: pip install msgpack-rpc-python
import numpy as np #pip install numpy
import msgpack
import math
import time
import sys
import os
import inspect
import types
import re
from airsim import MultirotorClient
import airsim


class MsgpackMixin:
    def __repr__(self):
        from pprint import pformat
        return "<" + type(self).__name__ + "> " + pformat(vars(self), indent=4, width=1)

    def to_msgpack(self, *args, **kwargs):
        return self.__dict__

    @classmethod
    def from_msgpack(cls, encoded):
        obj = cls()
        #obj.__dict__ = {k.decode('utf-8'): (from_msgpack(v.__class__, v) if hasattr(v, "__dict__") else v) for k, v in encoded.items()}
        obj.__dict__ = { k : (v if not isinstance(v, dict) else getattr(getattr(obj, k).__class__, "from_msgpack")(v)) for k, v in encoded.items()}
        #return cls(**msgpack.unpack(encoded))
        return obj


'''class MsgpackMixin:
    def to_msgpack(self, *args, **kwargs):
        return self.__dict__ #msgpack.dump(self.to_dict(*args, **kwargs))

    @classmethod
    def from_msgpack(cls, encoded):
        obj = cls()
        obj.__dict__ = {k.decode('utf-8'): v for k, v in encoded.items()}
        return obj'''''


class Vector3r(MsgpackMixin):
    x_val = np.float32(0)
    y_val = np.float32(0)
    z_val = np.float32(0)

    def __init__(self, x_val=np.float32(0), y_val=np.float32(0), z_val=np.float32(0)):
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val
    @staticmethod
    def nanVector3r():
        return Vector3r(np.nan, np.nan, np.nan)

    def containsNan(self):
        return (math.isnan(self.x_val) or math.isnan(self.y_val) or math.isnan(self.z_val))

    def __add__(self, other):
        return Vector3r(self.x_val + other.x_val, self.y_val + other.y_val, self.z_val + other.z_val)

    def __sub__(self, other):
        return Vector3r(self.x_val - other.x_val, self.y_val - other.y_val, self.z_val - other.z_val)

    def __truediv__(self, other):
        if type(other) in [int, float] + np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float']:
            return Vector3r( self.x_val / other, self.y_val / other, self.z_val / other)
        else:
            raise TypeError('unsupported operand type(s) for /: %s and %s' % ( str(type(self)), str(type(other))) )

    def __mul__(self, other):
        if type(other) in [int, float] + np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float']:
            return Vector3r(self.x_val*other, self.y_val*other, self.z_val*other)
        else:
            raise TypeError('unsupported operand type(s) for *: %s and %s' % ( str(type(self)), str(type(other))) )

    def dot(self, other):
        if type(self) == type(other):
            return self.x_val*other.x_val + self.y_val*other.y_val + self.z_val*other.z_val
        else:
            raise TypeError('unsupported operand type(s) for \'dot\': %s and %s' % ( str(type(self)), str(type(other))) )

    def cross(self, other):
        if type(self) == type(other):
            cross_product = np.cross(self.to_numpy_array(), other.to_numpy_array())
            return Vector3r(cross_product[0], cross_product[1], cross_product[2])
        else:
            raise TypeError('unsupported operand type(s) for \'cross\': %s and %s' % ( str(type(self)), str(type(other))) )

    def get_length(self):
        return ( self.x_val**2 + self.y_val**2 + self.z_val**2 )**0.5

    def distance_to(self, other):
        return ( (self.x_val-other.x_val)**2 + (self.y_val-other.y_val)**2 + (self.z_val-other.z_val)**2 )**0.5

    def to_Quaternionr(self):
        return Quaternionr(self.x_val, self.y_val, self.z_val, 0)

    def to_numpy_array(self):
        return np.array([self.x_val, self.y_val, self.z_val], dtype=np.float32)

class Quaternionr(MsgpackMixin):
    w_val = np.float32(0)
    x_val = np.float32(0)
    y_val = np.float32(0)
    z_val = np.float32(0)

    def __init__(self, x_val = np.float32(0), y_val = np.float32(0), z_val = np.float32(0), w_val = np.float32(1)):
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val
        self.w_val = w_val

class KinematicsState(MsgpackMixin):
    position = Vector3r()
    orientation = Quaternionr()
    linear_velocity = Vector3r()
    angular_velocity = Vector3r()
    linear_acceleration = Vector3r()
    angular_acceleration = Vector3r()

class CollisionInfo(MsgpackMixin):
    has_collided = False
    normal = Vector3r()
    impact_point = Vector3r()
    position = Vector3r()
    penetration_depth = 0.0
    time_stamp = 0.0
    object_name = ""
    object_id = -1

class GeoPoint(MsgpackMixin):
    latitude = 0.0
    longitude = 0.0
    altitude = 0.0

class LandedState:
    Landed = 0
    Flying = 1
class RCData(MsgpackMixin):
    timestamp = 0
    pitch, roll, throttle, yaw = (0.0,)*4 #init 4 variable to 0.0
    switch1, switch2, switch3, switch4 = (0,)*4
    switch5, switch6, switch7, switch8 = (0,)*4
    is_initialized = False
    is_valid = False
    def __init__(self, timestamp = 0, pitch = 0.0, roll = 0.0, throttle = 0.0, yaw = 0.0, switch1 = 0,
                 switch2 = 0, switch3 = 0, switch4 = 0, switch5 = 0, switch6 = 0, switch7 = 0, switch8 = 0, is_initialized = False, is_valid = False):
        self.timestamp = timestamp
        self.pitch = pitch
        self.roll = roll
        self.throttle = throttle
        self.yaw = yaw
        self.switch1 = switch1
        self.switch2 = switch2
        self.switch3 = switch3
        self.switch4 = switch4
        self.switch5 = switch5
        self.switch6 = switch6
        self.switch7 = switch7
        self.switch8 = switch8
        self.is_initialized = is_initialized
        self.is_valid = is_valid
class MultirotorState(MsgpackMixin):
    collision = CollisionInfo()
    kinematics_estimated_orientation = Quaternionr() # KinematicsState().orientation
    kinematics_estimated_position = Vector3r() #KinematicsState().position
    kinematics_estimated = KinematicsState()

    gps_location = GeoPoint()
    timestamp = np.uint64(0)
    landed_state = LandedState.Landed
    rc_data = RCData()
    ready = False
    ready_message = ""
    can_arm = False







class AirSimImageType:    
    Scene = 0
    DepthPlanner = 1
    DepthPerspective = 2
    DepthVis = 3
    DisparityNormalized = 4
    Segmentation = 5
    SurfaceNormals = 6

class DrivetrainType:
    MaxDegreeOfFreedom = 0
    ForwardOnly = 1
    
class LandedState:
    Landed = 0
    Flying = 1






"""class Pose(MsgpackMixin):
    position = Vector3r()
    orientation = Quaternionr()

    def __init__(self, position_val, orientation_val):
        self.position = position_val
        self.orientation = orientation_val"""
        
class Pose(MsgpackMixin):
    position = Vector3r()
    orientation = Quaternionr()

    def __init__(self, position_val = None, orientation_val = None):
        position_val = position_val if position_val != None else Vector3r()
        orientation_val = orientation_val if orientation_val != None else Quaternionr()
        self.position = position_val
        self.orientation = orientation_val

    @staticmethod
    def nanPose():
        return Pose(Vector3r.nanVector3r(), Quaternionr.nanQuaternionr())

    def containsNan(self):
        return (self.position.containsNan() or self.orientation.containsNan())


class CollisionInfo(MsgpackMixin):
    has_collided = False
    normal = Vector3r()
    impact_point = Vector3r()
    position = Vector3r()
    penetration_depth = np.float32(0)
    time_stamp = np.float32(0)
    object_name = ""
    object_id = -1

class GeoPoint(MsgpackMixin):
    latitude = 0.0
    longitude = 0.0
    altitude = 0.0

class YawMode(MsgpackMixin):
    is_rate = True
    yaw_or_rate = 0.0
    def __init__(self, is_rate = True, yaw_or_rate = 0.0):
        self.is_rate = is_rate
        self.yaw_or_rate = yaw_or_rate

class ImageRequest(MsgpackMixin):
    camera_id = np.uint8(0)
    image_type = AirSimImageType.Scene
    pixels_as_float = True
    compress = False

    def __init__(self, camera_id, image_type, pixels_as_float = True, compress = False):
        self.camera_id = camera_id
        self.image_type = image_type
        self.pixels_as_float = pixels_as_float
        self.compress = compress


class ImageResponse(MsgpackMixin):
    image_data_uint8 = np.uint8(0)
    image_data_float = np.float32(0)
    camera_position = Vector3r()
    camera_orientation = Quaternionr()
    time_stamp = np.uint64(0)
    message = ''
    pixels_as_float = np.float32(0)
    compress = True
    width = 0
    height = 0
    image_type = AirSimImageType.Scene

class CarControls(MsgpackMixin):
    throttle = np.float32(0)
    steering = np.float32(0)
    brake = np.float32(0)
    handbrake = False
    is_manual_gear = False
    manual_gear = 0
    gear_immediate = True

    def set_throttle(self, throttle_val, forward):
        if (forward):
            is_manual_gear = False
            manual_gear = 0
            throttle = abs(throttle_val)
        else:
            is_manual_gear = False
            manual_gear = -1
            throttle = - abs(throttle_val)

class CarState(MsgpackMixin):
    speed = np.float32(0)
    gear = 0
    position = Vector3r()
    velocity = Vector3r()
    orientation = Quaternionr()

class AirSimClientBase:
    def __init__(self, ip = "", port = 41451, timeout_value = 3600):
        if (ip == ""):
            ip = "127.0.0.1"
        self.client = msgpackrpc.Client(msgpackrpc.Address(ip, port), timeout = timeout_value, pack_encoding = 'utf-8', unpack_encoding = 'utf-8')

    """def __init__(self, ip, port):
        self.client = msgpackrpc.Client(msgpackrpc.Address(ip, port), timeout = 3600)"""

    def ping(self):
        return self.client.call('ping')
    
    def reset(self):
        self.client.call('reset')
    def confirmConnection(self):
        """
        Checks state of connection every 1 sec and reports it in Console so user can see the progress for connection.
        """
        if self.ping():
            print("Connected!")
        else:
             print("Ping returned false!")
    """def confirmConnection(self):
        print('Waiting for connection: ', end='')
        
        home = self.getHomeGeoPoint()
        while ((home.latitude == 0 and home.longitude == 0 and home.altitude == 0) or
                math.isnan(home.latitude) or  math.isnan(home.longitude) or  math.isnan(home.altitude)):
            time.sleep(1)
            home = self.getHomeGeoPoint()
            print('X', end='')
        print('')"""

    def simGetGroundTruthKinematics(self, vehicle_name = ''):
        kinematics_state = self.client.call('simGetGroundTruthKinematics', vehicle_name)
        return KinematicsState.from_msgpack(kinematics_state)
    simGetGroundTruthKinematics.__annotations__ = {'return': KinematicsState}

    def getHomeGeoPoint(self):
        return GeoPoint.from_msgpack(self.client.call('getHomeGeoPoint'))


    def enableApiControl(self, is_enabled, vehicle_name = ''):
        """
        Enables or disables API control for vehicle corresponding to vehicle_name

        Args:
            is_enabled (bool): True to enable, False to disable API control
            vehicle_name (str, optional): Name of the vehicle to send this command to
        """
        self.client.call('enableApiControl', is_enabled, vehicle_name)


    # basic flight control
    """def enableApiControl(self, is_enabled):
        return self.client.call('enableApiControl', is_enabled)"""
    def isApiControlEnabled(self):
        return self.client.call('isApiControlEnabled')

    def simSetSegmentationObjectID(self, mesh_name, object_id, is_name_regex = False):
        return self.client.call('simSetSegmentationObjectID', mesh_name, object_id, is_name_regex)
    def simGetSegmentationObjectID(self, mesh_name):
        return self.client.call('simGetSegmentationObjectID', mesh_name)
    def simGetVehiclePose(self, vehicle_name=''):
        """
        Args:
            vehicle_name (str, optional): Name of the vehicle to get the Pose of

        Returns:
            Pose:
        """
        pose = self.client.call('simGetVehiclePose', vehicle_name)
        return Pose.from_msgpack(pose)
    # camera control
    # simGetImage returns compressed png in array of bytes
    # image_type uses one of the AirSimImageType members
    def simGetImage(self, camera_id, image_type, vehicle_name = ''):
        # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
        camera_id = str(camera_id)
        result = self.client.call('simGetImage', camera_id, image_type,vehicle_name)
        if (result == "" or result == "\0"):
            return None
        return result

    # camera control
    # simGetImage returns compressed png in array of bytes
    # image_type uses one of the AirSimImageType members
    def simGetImages(self, requests, vehicle_name = ''):
        """
        Get multiple images

        See https://microsoft.github.io/AirSim/image_apis/ for details and examples

        Args:
            requests (list[ImageRequest]): Images required
            vehicle_name (str, optional): Name of vehicle associated with the camera

        Returns:
            list[ImageResponse]:
        """
        responses_raw = self.client.call('simGetImages', requests, vehicle_name)
        return [ImageResponse.from_msgpack(response_raw) for response_raw in responses_raw]


    """"def getCollisionInfo(self):
        return CollisionInfo.from_msgpack(self.client.call('getCollisionInfo'))"""

    def simGetCollisionInfo(self, vehicle_name = ''):
        """
        Args:
            vehicle_name (str, optional): Name of the Vehicle to get the info of

        Returns:
            CollisionInfo:
        """
        return CollisionInfo.from_msgpack(self.client.call('simGetCollisionInfo', vehicle_name))

    @staticmethod
    def stringToUint8Array(bstr):
        return np.fromstring(bstr, np.uint8)
    @staticmethod
    def stringToFloatArray(bstr):
        return np.fromstring(bstr, np.float32)
    @staticmethod
    def listTo2DFloatArray(flst, width, height):
        return np.reshape(np.asarray(flst, np.float32), (height, width))
    @staticmethod
    def getPfmArray(response):
        return AirSimClientBase.listTo2DFloatArray(response.image_data_float, response.width, response.height)

    @staticmethod
    def get_public_fields(obj):
        return [attr for attr in dir(obj)
                             if not (attr.startswith("_") 
                                or inspect.isbuiltin(attr)
                                or inspect.isfunction(attr)
                                or inspect.ismethod(attr))]


    @staticmethod
    def to_dict(obj):
        return dict([attr, getattr(obj, attr)] for attr in AirSimClientBase.get_public_fields(obj))

    @staticmethod
    def to_str(obj):
        return str(AirSimClientBase.to_dict(obj))

    @staticmethod
    def write_file(filename, bstr):
        with open(filename, 'wb') as afile:
            afile.write(bstr)



    def simSetPose(self, pose, ignore_collison):
        self.client.call('simSetPose', pose, ignore_collison)

    def simGetPose(self):
        return self.client.call('simGetPose')
    def simSetVehiclePose(self, pose, ignore_collison, vehicle_name = ''):
        """
        Set the pose of the vehicle
        If you don't want to change position (or orientation) then just set components of position (or orientation) to floating point nan values
        Args:
            pose (Pose): Desired Pose pf the vehicle
            ignore_collision (bool): Whether to ignore any collision or not
            vehicle_name (str, optional): Name of the vehicle to move
        """
        self.client.call('simSetVehiclePose', pose, ignore_collison, vehicle_name)

    def simGetVehiclePose(self, vehicle_name = ''):
        """
        Args:
            vehicle_name (str, optional): Name of the vehicle to get the Pose of
        Returns:
            Pose:
        """
        pose = self.client.call('simGetVehiclePose', vehicle_name)
        return Pose.from_msgpack(pose)

    # helper method for converting getOrientation to roll/pitch/yaw
    # https:#en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    @staticmethod
    def toEulerianAngle(q):
        z = q.z_val
        y = q.y_val
        x = q.x_val
        w = q.w_val
        ysqr = y * y

        # roll (x-axis rotation)
        t0 = +2.0 * (w*x + y*z)
        t1 = +1.0 - 2.0*(x*x + ysqr)
        roll = math.atan2(t0, t1)

        # pitch (y-axis rotation)
        t2 = +2.0 * (w*y - z*x)
        if (t2 > 1.0):
            t2 = 1
        if (t2 < -1.0):
            t2 = -1.0
        pitch = math.asin(t2)

        # yaw (z-axis rotation)
        t3 = +2.0 * (w*z + x*y)
        t4 = +1.0 - 2.0 * (ysqr + z*z)
        yaw = math.atan2(t3, t4)

        return (pitch, roll, yaw)

    @staticmethod
    def toQuaternion(pitch, roll, yaw):
        t0 = math.cos(yaw * 0.5)
        t1 = math.sin(yaw * 0.5)
        t2 = math.cos(roll * 0.5)
        t3 = math.sin(roll * 0.5)
        t4 = math.cos(pitch * 0.5)
        t5 = math.sin(pitch * 0.5)

        q = Quaternionr()
        q.w_val = t0 * t2 * t4 + t1 * t3 * t5 #w
        q.x_val = t0 * t3 * t4 - t1 * t2 * t5 #x
        q.y_val = t0 * t2 * t5 + t1 * t3 * t4 #y
        q.z_val = t1 * t2 * t4 - t0 * t3 * t5 #z
        return q

    @staticmethod
    def wait_key(message = ''):
        ''' Wait for a key press on the console and return it. '''
        if message != '':
            print (message)

        result = None
        if os.name == 'nt':
            import msvcrt
            result = msvcrt.getch()
        else:
            import termios
            fd = sys.stdin.fileno()

            oldterm = termios.tcgetattr(fd)
            newattr = termios.tcgetattr(fd)
            newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, newattr)

            try:
                result = sys.stdin.read(1)
            except IOError:
                pass
            finally:
                termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)

        return result

    @staticmethod
    def read_pfm(file):
        """ Read a pfm file """
        file = open(file, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        header = str(bytes.decode(header, encoding='utf-8'))
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        temp_str = str(bytes.decode(file.readline(), encoding='utf-8'))
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', temp_str)
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        # DEY: I don't know why this was there.
        #data = np.flipud(data)
        file.close()
    
        return data, scale

    @staticmethod
    def write_pfm(file, image, scale=1):
        """ Write a pfm file """
        file = open(file, 'wb')

        color = None

        if image.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3: # color image
            color = True
        elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
            color = False
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        file.write('PF\n'.encode('utf-8')  if color else 'Pf\n'.encode('utf-8'))
        temp_str = '%d %d\n' % (image.shape[1], image.shape[0])
        file.write(temp_str.encode('utf-8'))

        endian = image.dtype.byteorder

        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        temp_str = '%f\n' % scale
        file.write(temp_str.encode('utf-8'))

        image.tofile(file)

    @staticmethod
    def write_png(filename, image):
        """ image must be numpy array H X W X channels
        """
        import zlib, struct

        buf = image.flatten().tobytes()
        width = image.shape[1]
        height = image.shape[0]

        # reverse the vertical line order and add null bytes at the start
        width_byte_4 = width * 4
        raw_data = b''.join(b'\x00' + buf[span:span + width_byte_4]
                            for span in range((height - 1) * width_byte_4, -1, - width_byte_4))

        def png_pack(png_tag, data):
            chunk_head = png_tag + data
            return (struct.pack("!I", len(data)) +
                    chunk_head +
                    struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))

        png_bytes = b''.join([
            b'\x89PNG\r\n\x1a\n',
            png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
            png_pack(b'IDAT', zlib.compress(raw_data, 9)),
            png_pack(b'IEND', b'')])

        AirSimClientBase.write_file(filename, png_bytes)


# -----------------------------------  Multirotor APIs ---------------------------------------------
class MultirotorClient(AirSimClientBase, object):
    def __init__(self, ip = "", port = 41451, timeout_value = 3600):
        super(MultirotorClient, self).__init__(ip, port, timeout_value)

    """def __init__(self, ip = ""):
        if (ip == ""):
            ip = "127.0.0.1"
        super(MultirotorClient, self).__init__(ip, 41451)"""

    '''def armDisarm(self, arm):
        return self.client.call('armDisarm', arm)'''
    def armDisarm(self, arm, vehicle_name = ''):
        """
        Arms or disarms vehicle

        Args:
            arm (bool): True to arm, False to disarm the vehicle
            vehicle_name (str, optional): Name of the vehicle to send this command to

        Returns:
            bool: Success
        """
        return self.client.call('armDisarm', arm, vehicle_name)

    def takeoff(self, max_wait_seconds = 15):
        return self.client.call('takeoff', max_wait_seconds)

    def takeoffAsync(self, timeout_sec=20, vehicle_name=''):
        """
        Takeoff vehicle to 3m above ground. Vehicle should not be moving when this API is used
        Args:
            timeout_sec (int, optional): Timeout for the vehicle to reach desired altitude
            vehicle_name (str, optional): Name of the vehicle to send this command to
        Returns:
            msgpackrpc.future.Future: future. call .join() to wait for method to finish. Example: client.METHOD().join()
        """
        return self.client.call_async('takeoff', timeout_sec, vehicle_name)
        
    def land(self, max_wait_seconds = 60):
        return self.client.call('land', max_wait_seconds)
        
    def goHome(self):
        return self.client.call('goHome')

    def hover(self):
        return self.client.call('hover')




    # query vehicle state
    def getPosition(self):
        return Vector3r.from_msgpack(self.client.call('getPosition'))

    def getVelocity(self):
        return Vector3r.from_msgpack(self.client.call('getVelocity'))

    def getOrientation(self):
        return Quaternionr.from_msgpack(self.client.call('getOrientation'))

    def getLandedState(self):
        return self.client.call('getLandedState')
    def getGpsLocation(self):
        return GeoPoint.from_msgpack(self.client.call('getGpsLocation'))
    def getPitchRollYaw(self):
        return self.toEulerianAngle(self.getOrientation())


    #def getRCData(self):
    #    return self.client.call('getRCData')
    def timestampNow(self):
        return self.client.call('timestampNow')
    def isApiControlEnabled(self):
        return self.client.call('isApiControlEnabled')
    def isSimulationMode(self):
        return self.client.call('isSimulationMode')
    def getServerDebugInfo(self):
        return self.client.call('getServerDebugInfo')


    # APIs for control
    def moveByAngle(self, pitch, roll, z, yaw, duration):
        return self.client.call('moveByAngle', pitch, roll, z, yaw, duration)


    def moveByVelocityAsync(self, vx, vy, vz, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), vehicle_name = ''):
        """
        Args:
            vx (float): desired velocity in world (NED) X axis
            vy (float): desired velocity in world (NED) Y axis
            vz (float): desired velocity in world (NED) Z axis
            duration (float): Desired amount of time (seconds), to send this command for
            drivetrain (DrivetrainType, optional):
            yaw_mode (YawMode, optional):
            vehicle_name (str, optional): Name of the multirotor to send this command to

        Returns:
            msgpackrpc.future.Future: future. call .join() to wait for method to finish. Example: client.METHOD().join()
        """
        return self.client.call_async('moveByVelocity', vx, vy, vz, duration, drivetrain, yaw_mode, vehicle_name)

    """def moveByVelocityZ(self, vx, vy, z, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode()):
        return self.client.call('moveByVelocityZ', vx, vy, z, duration, drivetrain, yaw_mode)"""
    def moveByVelocityZAsync(self, vx, vy, z, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), vehicle_name = ''):
        return self.client.call_async('moveByVelocityZ', vx, vy, z, duration, drivetrain, yaw_mode, vehicle_name)
    def moveOnPath(self, path, velocity, max_wait_seconds = 60, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), lookahead = -1, adaptive_lookahead = 1):
        return self.client.call('moveOnPath', path, velocity, max_wait_seconds, drivetrain, yaw_mode, lookahead, adaptive_lookahead)

    """def moveToZ(self, z, velocity, max_wait_seconds = 60, yaw_mode = YawMode(), lookahead = -1, adaptive_lookahead = 1):
        return self.client.call('moveToZ', z, velocity, max_wait_seconds, yaw_mode, lookahead, adaptive_lookahead)"""

    def moveToZAsync(self, z, velocity, timeout_sec=3e+38, yaw_mode=YawMode(), lookahead=-1, adaptive_lookahead=1,vehicle_name=''):
        return self.client.call_async('moveToZ', z, velocity, timeout_sec, yaw_mode, lookahead, adaptive_lookahead,vehicle_name)
    def moveToPosition(self, x, y, z, velocity, max_wait_seconds = 60, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), lookahead = -1, adaptive_lookahead = 1):
        return self.client.call('moveToPosition', x, y, z, velocity, max_wait_seconds, drivetrain, yaw_mode, lookahead, adaptive_lookahead)

    def moveByManual(self, vx_max, vy_max, z_min, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode()):
        return self.client.call('moveByManual', vx_max, vy_max, z_min, duration, drivetrain, yaw_mode)

    def rotateToYaw(self, yaw, max_wait_seconds = 60, margin = 5):
        return self.client.call('rotateToYaw', yaw, max_wait_seconds, margin)


    def rotateByYawRateAsync(self, yaw_rate, duration, vehicle_name = ''):
        return self.client.call_async('rotateByYawRate', yaw_rate, duration, vehicle_name)

    def getMultirotorState(self, vehicle_name=''):
        return MultirotorState.from_msgpack(self.client.call('getMultirotorState', vehicle_name))

    getMultirotorState.__annotations__ = {'return': MultirotorState}

# -----------------------------------  Car APIs ---------------------------------------------
class CarClient(AirSimClientBase, object):
    def __init__(self, ip = ""):
        if (ip == ""):
            ip = "127.0.0.1"
        super(CarClient, self).__init__(ip, 42451)

    def setCarControls(self, controls):
        self.client.call('setCarControls', controls)

    def getCarState(self):
        state_raw = self.client.call('getCarState')
        return CarState.from_msgpack(state_raw)
