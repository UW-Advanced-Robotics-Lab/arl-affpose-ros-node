#!/usr/bin/env python
# coding=utf8

import cv2
import numpy as np

import rospy

#######################################
#######################################

class KalmanFilter():
    def __init__(self,
                 num_states=6,
                 num_measurements=6,
                 initial_covariance=0.1,
                 process_noise=1e-5,
                 measurement_noise=1e-4,
                 dt=1/15):

        # num states
        self.num_states = num_states                # object [x, y, z, roll, pitch, yaw]
        self.num_measurements = num_measurements    # object pose in camera frame

        self.initial_covariance = initial_covariance

        # As the measurement noise is reduced the faster will converge
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Run-time FPS
        self.dt = dt

        # need an estimate of the state to begin filtering
        self._is_init = False

    ######################################
    ######################################

    def initialize(self, x_in):

        self.x_est = x_in
        # initial guess of state covariance
        self.p_est = np.eye(self.num_states) * self.initial_covariance

        # motion model is static (i.e. object's don't move)
        self.A = np.eye(self.num_states)
        # we do not apply any control actions
        self.B = np.zeros(self.num_states)
        # state process covariance matrix
        self.R = np.eye(self.num_states) * self.process_noise

        # measurement model is [x, y, z, roll, pitch, yaw]
        self.C = np.eye(self.num_measurements)
        # process covariance matrix
        self.Q = np.eye(self.num_measurements) * self.measurement_noise

        self._is_init = True

    #######################################
    #######################################

    def prediction(self):

        # assuming object(s) remains static
        # x_est = A*x_(t-1) + B*u_(t-1)
        self.x_est = self.x_est # np.dot(self.A, self.x_est)

        # updating covariance
        # p_est = A * P_(t-1) * A^T + R
        self.p_est = self.p_est # np.dot(np.dot(self.A, self.p_est), self.A.transpose()) + self.R

    def correction(self, measurement):

        # updating how much we should trust measurement - or Kalman Gain, K
        # K = p_est * C^T * (C * p_est * C^T + Q)^-1
        # self.K = np.dot(np.dot(self.p_est, self.C.transpose()), np.linalg.inv(np.dot(np.dot(self.C, self.p_est), self.C.transpose()) + self.Q))
        self.K = np.dot(self.p_est, np.linalg.inv(self.p_est + self.Q))

        # updating state
        self.x_est += np.dot(self.K, measurement - self.x_est)

        # updating covariance
        self.p_est = np.dot((np.eye(self.num_states) - self.K), self.p_est)

        return self.x_est

    #######################################
    #######################################
