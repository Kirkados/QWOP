# QWOP
This project aims to use deep reinforcement learning to play the game [QWOP](http://www.foddy.net/Athletics.html). 

It is the first in a series of collaboration projects between [PTStephD](https://github.com/PTStephD) and [Kirkados](https://github.com/Kirkados)

The Algorithm
=============
The core deep reinforcement learning algorithm is the Distributional Deep Q Learning algorithm, first presented by [Bellmare et al. in 2017](https://arxiv.org/pdf/1707.06887). A number of enhancements developed by other researchers are used as well. Namely:
+ [Parallel actors and learners](https://arxiv.org/pdf/1602.01783)
+ [N-step returns](https://arxiv.org/pdf/1602.01783)
+ [Prioritized experience replay](http://arxiv.org/abs/1511.05952)
The algorithm is written in Tensorflow 1.15.

Special thanks to:
+ [msinto93] (https://github.com/msinto93)
+ [SuReLI]   (https://github.com/SuReLI)
+ [DeepMind] (https://github.com/deepmind)
+ [OpenAI]   (https://github.com/openai)

for publishing their codes! The open-source mindset of AI research is fantastic.

Results
-----
Incentivizing the agent to run down the track (positive rewards are given for forward velocity): https://youtu.be/OYBiUWuA4Ho

Incentivizing the agent to run down the track AND perform front flips: https://youtu.be/16JEWNf6468


Usage
-----
To run the training algorithm, edit `settings.py` and `environment_qwop_full_11` as appropriate, and then run
`python3 main.py` from a terminal.
In addition to python, the following python3 packages must be installed:
+ psutil `pip3 install psutil`
+ Tensorflow `pip3 install tensorflow` or `pip3 install tensorflow-gpu` for GPU compatibility (Additional steps required)
+ box2d `pip3 install box2d-py`
+ matplotlib `pip3 install matplotlib`
+ OpenAI gym `pip3 install gym[all]`
+ virtual display `pip3 install pyvirtualdisplay`
The following linux packages must also be installed:
+ Opengl `sudo apt-get install python-opengl`
+ xvfb `sudo apt-get install xvfb`
+ ffmpeg `sudo apt-get install ffmpeg`

The Environment
===============
A QWOP dynamics environment was developed from first principles and is contained in `environment_qwop_full_11.py`. It consists of a stick figure with a torso, two arms, and two legs. The goal is to press the buttons `Q`, `W`, `O`, and `P` to make the stick figure translate down the track as fast as possible.
