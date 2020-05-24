#!/usr/bin/env bash
loss_full_cpy_weight=0.4 ./run.py -g 0 -c config.non_gan -n non_gan_fullcpyloss_0.4 &
loss_full_cpy_weight=0.5 ./run.py -g 1 -c config.non_gan -n non_gan_fullcpyloss_0.5 &
loss_full_cpy_weight=0.6 ./run.py -g 2 -c config.non_gan -n non_gan_fullcpyloss_0.6 &
loss_full_cpy_weight=0.7 ./run.py -g 3 -c config.non_gan -n non_gan_fullcpyloss_0.7 &
