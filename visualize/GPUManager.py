#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:13:40 2017

@author: diegothomas
"""

import pyopencl as cl
import imp
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
KernelsOpenCL = imp.load_source('KernelsOpenCL', './KernelsOpenCL.py')

    
class GPUManager():
    """
    GPUManager is the class that initialized GPU envirenment and functions

    Work-item: the basic unit of work on an OpenCL device
    Kernel: the code for a work-item (basically a C function)
    Program: Collection of kernels and other functions (analogous to a dynamic library)
    Context: The environment within which workitems execute; includes devices and their memories and command queues
    Command Queue: A queue used by the Host application to submit work to a Device (e.g., kernel execution instances)
                    Namely, it is control work order.
       – Work is queued in-order, one queue per device
       – Work can be executed in-order or out-of-order
    Platform: The host plus a collection of devices managed by the OpenCL framework that allow an application to share resources and execute kernels on devices in the platform.
    """

    def __init__(self):
        # """
        # Constructor
        # """
        # self.platform = cl.get_platforms()[0]
        # self.devices = self.platform.get_devices()
        # # select the device you want to work with
        # self.context = cl.Context([self.devices[0]])
        # #self.context = cl.Context([self.devices[1]])
        # self.queue = cl.CommandQueue(self.context)
        # self.programs = {}
        pass

    def print_device_info(self):
        """
        Display information on selected devices
        :return:  none
        """
        print ('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')
        print('=' * 60)
        print('Platform - Name: ' + self.platform.name)
        print('Platform - Vendor: ' + self.platform.vendor)
        print('Platform - Version: ' + self.platform.version)
        print('Platform - Profile: ' + self.platform.profile)
        
        for device in self.devices:
            print('   ' + '-' * 56)
            print('   Device - Name:  ' + device.name)
            print('   Device - Type:  ' + cl.device_type.to_string(device.type))
            print('   Device - Max Clock Speed:  {0} MHz' .format(device.max_clock_frequency))
            print('   Device - Compute Units:  {0}' .format(device.max_compute_units))
            print('   Device - Local Memory:  {0:.0f} KB' .format(device.local_mem_size/1024.0))
            print('   Device - Constant Memory:  {0:.0f} KB' .format(device.max_constant_buffer_size/1024.0))
            print('   Device - Global Memory:  {0:.0f} GB' .format(device.global_mem_size/1073741824.0))
            print('   Device - Max Buffer/Image Size:  {0:.0f} MB' .format(device.max_mem_alloc_size/1048567.0))
            print('   Device - Max Work Group Size:  {0:.0f}' .format(device.max_work_group_size))
        print('\n')
        

    def load_kernels(self):
        """
        Load programs with its kernels
        :return: none
        """

        self.programs['FuseSortedTSDF'] = cl.Program(self.context, KernelsOpenCL.Kernel_FuseTSDF).build()
        #self.programs['FuseEqTSDF'] = cl.Program(self.context, KernelsOpenCL.Kernel_FuseEqTSDF).build()
        #self.programs['Test'] = cl.Program(self.context, KernelsOpenCL.Kernel_Test).build()
        
