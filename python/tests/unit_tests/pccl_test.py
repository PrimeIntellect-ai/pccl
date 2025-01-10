from time import sleep

import pytest
import torch

def test_import_lib():
    import pccl

from pccl import *

def test_master_node_run():
    m = MasterNode(listen_address='127.0.0.1:48148')
    m.run()
    m.interrupt()

def test_communicator():
    m = MasterNode(listen_address='127.0.0.1:48148')
    m.run()
    c = Communicator('127.0.0.1:48148', 0)
    c.connect()
    m.interrupt()

def test_communicator_destructor_with_connect():
    m = MasterNode(listen_address='127.0.0.1:48148')
    m.run()

    def connect():
        c = Communicator('127.0.0.1:48148', 0)
        c.connect()

    connect()
    m.interrupt()

def test_communicator_destructor_without_connect():
    m = MasterNode(listen_address='127.0.0.1:48148')
    m.run()

    def connect():
        c = Communicator('127.0.0.1:48148', 0)
    connect()
    m.interrupt()

def test_communicator_update_topology_without_connect():
    m = MasterNode(listen_address='127.0.0.1:48148')
    m.run()
    c = Communicator('127.0.0.1:48148', 0)
    with pytest.raises(PCCLError):
        c.update_topology()
    m.interrupt()
