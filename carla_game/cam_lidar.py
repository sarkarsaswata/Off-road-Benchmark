# coding: utf-8
#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import random
import weakref
from carla_game.waypoints.waypoints import *
from common.utills import numpy_imwrite
import cv2

try:
    pwd = os.getcwd()
    search_key = '%s/carla_v09/dist/carla-*%d.*-%s.egg' % (
        pwd,
        sys.version_info.major,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'
    )
    carla_path = glob.glob(search_key)[0]
    sys.path.append(carla_path)
    
    import carla
    from carla import ColorConverter as cc    
except IndexError:
    raise RuntimeError('cannot find carla directory')

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

try:
    import torch
except ImportError:
    raise RuntimeError('cannot import pytorch, make sure pytorch package is installed')

import subprocess
import time
import signal
import collections

# Starting location of the rough terrain city
ENV_INIT_POS = {
    "Offroad_1": [{"pos":(8770.0/100, 5810.0/100, 120.0/100), "yaw":-93.0},
                  {"pos":(-660.0/100, 260.0/100, 120.0/100), "yaw":-195.0}],

    "Offroad_2": [{"pos":(1774.0/100, 4825.0/100, 583.0/100), "yaw":92.0},
                  {"pos":(20417.0/100, 15631.0/100, 411.0/100), "yaw":-55.0}],

    "Offroad_3": [{"pos":(26407.0/100, -4893.0/100, 181.0/100), "yaw":-90.0},
                  {"pos":(-13270.0/100, 3264.0/100, 124.0/100), "yaw":-38.0}],

    "Offroad_4": [{"pos":(-12860.0/100, 22770.0/100, 210.0/100), "yaw":0.0}],

    "Offroad_5": [{"pos":(4738/100, 7365/100, 131/100), "yaw":90},
                  {"pos":(12599/100, 3244/100, 951/100), "yaw":224},
                  {"pos":(6203/100, -12706/100, 214/100), "yaw":69},
                  {"pos":(-9863/100, -13876/100, 388/100), "yaw":162}],

    "Offroad_6": [{"pos":(-12433/100, -11850/100, 99/100), "yaw":126},
                  {"pos":(-2977/100, -11090/100, 548/100), "yaw":66},
                  {"pos":(5308/100, -1616/100, 464/100), "yaw":52},
                  {"pos":(15605/100, 1639/100, 610/100), "yaw":68}],

    "Offroad_7": [{"pos":(-10564/100, -15334/100, 107/100), "yaw":166},
                  {"pos":(-4962/100, 651/100, 276/100), "yaw":134},
                  {"pos":(8814/100, -6371/100, 210/100), "yaw":121},
                  {"pos":(13450/100, 5045/100, 534/100), "yaw":145}],

    "Offroad_8": [{"pos":(-11109/100, 7814/100, 366/100), "yaw":57},
                  {"pos":(4453/100, 14202/100, 300/100), "yaw":-11},
                  {"pos":(11333/100, -14194/100, 483/100), "yaw":45},
                  {"pos":(-4416/100, -4294/100, 158/100), "yaw":210}],

    "Track1": [{"pos":(6187.0/100, 6686.0/100, 138.0/100), "yaw":-91.0}],

    "Town01": [{"pos":(8850.0/100, 9470.0/100, 100.0/100), "yaw":90.0},
               {"pos":(33880.0/100, 24380.0/100, 100.0/100), "yaw":-90.0}],

    "Town02": [{"pos":(8760.0/100, 18760.0/100, 100.0/100), "yaw":-0.0},
               {"pos":(10850.0/100, 30690.0/100, 100.0/100), "yaw":0.0}]
}

ENV_NAME = ENV_INIT_POS.keys()

weatherS = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "CloudySunset": carla.WeatherParameters.CloudySunset,
    "WetSunset": carla.WeatherParameters.WetSunset,
    "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
    "MidRainSunset": carla.WeatherParameters.MidRainSunset,
    "HardRainSunset": carla.WeatherParameters.HardRainSunset,
    "SoftRainSunset": carla.WeatherParameters.SoftRainSunset
}


class CarlaEnv(object):
    """
    CarlaEnv parameters
        log_dir : save log data path (must write)
        data_dir : data path (Not use)
        host : client host name (default: localhost)
        port : client port number (default: 2000)
        server_path : server`s absolute path (default: CARLA_ROOT)
        server_size : size of server image (default: 400,300)
        image_size : size of client(pygame) image (default: 800,600)
        city_name : name of city name (choose Offroad_1, Offroad_2, Offroad_3, Offroad_4, Track) (default: Offroad_2)
        render : choose render image or not(pygame and server) (default: True)
    """

    city_lenghts = {
        "Offroad_1": 1200,
        "Offroad_2": 670,
        "Offroad_3": 3800,
        "Offroad_4": 7000,
        "Offroad_5": 1800,
        "Offroad_6": 1830,
        "Offroad_7": 2020,
        "Offroad_8": 2390
    }

    def __init__(self,
                 log_dir,
                 data_dir=None,
                 host='localhost',
                 port=2000,
                 server_path=None,
                 server_size=(400, 300),
                 image_size=(800,600),
                 fps=10,
                 city_name='Offroad_2',
                 weather='ClearNoon',
                 render = True,
                 render_gcam = False,
                 gcam_target_layer = "conv_layer.1.res2",
                 gcam_target_model = "IL",
                 plot = True,
                 is_image_state = True):
        
        self.frame = None
        self.log_dir = log_dir
        self.data_dir = data_dir
        self.delta_seconds = 1.0 / fps
        self.fps = fps
        self.host = host
        self.port = port
        self.server_size = server_size
        self.image_size = image_size
        self.city_name = city_name
        self._render = render
        self.gamma_correction = 2.2
        self.speed_up_steps = 10
        self.timeout_step = 100000
        self._server_path = str(server_path)
        self.is_image_state = is_image_state
        self._settings = None
        self._queues = []
        self._history = []
        self.server = None
        self.server_pid = -99999
        self.client = None  # make client
        self.world = None # connect client world
        self.setup_client_and_server()
        self.waypoints_manager = get_waypoints_manager(city_name)
        self.world.set_weather(weatherS[weather])
        self.weather = weather

        self.bp = self.world.get_blueprint_library() # blueprint library

        self.vehicle = None
        self._control = None
        self.sensors = [] # Image sensor
        self.collision_sensor = None

        self.set_position()

        self._plot = plot
        if plot:
            self.plotter = Animator(lims=[
                self.waypoints_manager.total_min-10,
                self.waypoints_manager.total_max+10
            ])

        self._render_gcam = render_gcam
        # for rendering
        if self._render:
            self.display = None
            self.clock = None
            self.init_for_render()

            if self._render_gcam:
                from main_grad_cam import Gcam_generator
                self.gcam_generator = Gcam_generator(gcam_target_layer, gcam_target_model, cuda=True)

        self._record = False

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        
        for sensor in self.sensors:
            make_queue(sensor.listen)
        
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(lambda event: CarlaEnv._on_collision(weak_self, event))
        return self
    
    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)
    
    def _open_server(self):
        with open(self.log_dir, "wb") as out:
            cmd = [os.path.join(os.environ.get('CARLA_ROOT'), 'CarlaUE4.sh'),
            self.city_name, "-carla-server", "-fps=10", "-world-port={}".format(self.port),
            "-windwed -ResX={} -ResY={}".format(self.server_size[0],self.server_size[1])
            ]
            
            p = subprocess.Popen(cmd, stdout=out, stderr=out)
            time.sleep(5)
        return p
    
    def _close_server(self):
        no_of_attemps = 0
        try:
            while self.is_process_alive(self.server_pid):
                print("Trying to close Carla server with pid %d" % self.server_pid)
                if no_of_attemps < 5:
                    self.server.terminate()
                elif no_of_attemps < 10:
                    self.server.kill()
                else:
                    os.kill(self.server_pid, signal.SIGKILL)
                time.sleep(1)
                no_of_attemps += 1
        except Exception as e:
            print(e)

    def is_process_alive(self, pid):
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def set_position(self):
        pos, paths = self.waypoints_manager.get_init_pos()
        self.init_paths = paths
        self.init_pos = {
            "pos": (pos[0], pos[1], pos[2] + 2),
            "yaw": pos[3]
        }
        self.init_state = carla.Transform(carla.Location(*self.init_pos["pos"]),
                                          carla.Rotation(yaw=self.init_pos["yaw"]))
