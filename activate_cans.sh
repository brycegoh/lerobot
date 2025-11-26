#!/bin/bash

BITRATE=1000000


sudo ip link set can0 down
sudo ip link set can1 down

# Set bitrate and bring interfaces up
sudo ip link set can0 up type can bitrate $BITRATE
sudo ip link set can1 up type can bitrate $BITRATE

# Show status
ip -details link show can0
ip -details link show can1