#!/usr/bin/env python2

import numpy as np

TEMPLATE = """<mujoco model="ant">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <custom>
    <numeric data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
  </custom>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 0.75">
      <geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
      <!--
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootyaw" pos="0 0 0" stiffness="0" type="hinge"/>
      -->
      <body name="front_left_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="aux_1_geom" size="0.08" type="capsule"/>
        <body name="aux_1" pos="0.2 0.2 0">
          <joint axis="0 0 1" name="hip_1" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="left_leg_geom" size="0.08" type="capsule"/>
          <body pos="0.2 0.2 0">
            <joint axis="-1 1 0" name="ankle_1" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 0.4 0.4 0.0" name="left_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="front_right_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="aux_2_geom" size="0.08" type="capsule"/>
        <body name="aux_2" pos="-0.2 0.2 0">
          <joint axis="0 0 1" name="hip_2" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="right_leg_geom" size="0.08" type="capsule"/>
          <body pos="-0.2 0.2 0">
            <joint axis="1 1 0" name="ankle_2" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -0.4 0.4 0.0" name="right_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="aux_3_geom" size="0.08" type="capsule"/>
        <body name="aux_3" pos="-0.2 -0.2 0">
          <joint axis="0 0 1" name="hip_3" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="back_leg_geom" size="0.08" type="capsule"/>
          <body pos="-0.2 -0.2 0">
            <joint axis="-1 1 0" name="ankle_3" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -0.4 -0.4 0.0" name="third_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="right_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 0.2 -0.2 0.0" name="aux_4_geom" size="0.08" type="capsule"/>
        <body name="aux_4" pos="0.2 -0.2 0">
          <joint axis="0 0 1" name="hip_4" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 0.2 -0.2 0.0" name="rightback_leg_geom" size="0.08" type="capsule"/>
          <body pos="0.2 -0.2 0">
            <joint axis="1 1 0" name="ankle_4" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 0.4 -0.4 0.0" name="fourth_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
    </body>

    %s

  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_4" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_1" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_2" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_3" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_3" gear="150"/>
  </actuator>
</mujoco>"""

WALL = """
    <body name="%s" pos="%f 0 0">
      <geom size="0.05 0.4 0.1" density="0.00001" contype="1" conaffinity="1" type="box" rgba="0.5 0.5 0.5 1"/>
    </body>
"""

#FLOOR = """
#    <body name="%s" pos="%f %f 0">
#      <geom size="1.5 1.5 0.01" type="box" conaffinity="0" contype="0" rgba="%s"/>
#    </body>
#"""
FLOOR = """
    <body name="%s" pos="%f %f 0">
      <geom size="1 1 0.01" type="box" conaffinity="0" contype="0" rgba="%s"/>
    </body>
"""

body_counter = 1
colors = ["0.5 0 0 0.75", "0 0.5 0 0.75", "0 0 0.5 0.75"]
for i in range(50):
    while True:
        red_pos, green_pos, blue_pos = [
                tuple(p) for p in (np.random.randint(5, size=(3, 2)) - 2) * 3]
        if len({red_pos, green_pos, blue_pos}) == 3:
            break

    bodies = []
    for (x, y), color in zip([red_pos, green_pos, blue_pos], colors):
        name = "floor_%d" % body_counter
        bodies.append(FLOOR % (name, x, y, color))
        body_counter += 1

    i_color = np.random.choice(3)
    if i < 40:
        i_offset = np.random.choice(3)
        if i_offset == i_color:
            i_offset += 1
    else:
        i_offset = i_color

    goal_x, goal_y = [red_pos, green_pos, blue_pos][i_color]
    if i_offset == 0:
        goal_y += 3
    elif i_offset == 1:
        goal_x += 3
    elif i_offset == 2:
        goal_y -= 3
    elif i_offset == 3:
        goal_x -= 3

    name = "target_%d" % body_counter
    bodies.append(FLOOR % (name, goal_x, goal_y, "1 1 1 0.5"))
    body_counter += 1

    with open("map_%d.xml" % i, "w") as map_f:
        print >>map_f, TEMPLATE % "\n".join(bodies)

    with open("data_%d.txt" % i, "w") as data_f:
        for x, y in [red_pos, green_pos, blue_pos]:
            print >>data_f, x, y
        print >>data_f, goal_x, goal_y
        print >>data_f, ["red", "green", "blue"][i_color],
        print >>data_f, ["north", "east", "south", "west"][i_offset]

#body_counter = 1
#for spec in SPECS:
#    x = 0
#    y = 0
#    dirs = spec.split(",")
#    bodies = []
#    bodies.append(FLOOR % ("floor_0", x, y, "0.2 0.2 0.5 0.7"))
#    last_dir = None
#    for d in dirs:
#        nx, ny = x, y
#        if d == "north":
#            ny += 3
#            color = "0.2 0.2 0.5 0.7"
#        elif d == "east":
#            nx += 3
#            color = "0.2 0.2 0.5 0.7"
#        elif d == "south":
#            ny -= 3
#            color = "0.2 0.2 0.5 0.7"
#        elif d == "west":
#            nx -= 3
#            color = "0.2 0.2 0.5 0.7"
#
#        name = "floor_%d" % body_counter
#        bodies.append(FLOOR % (name, nx, ny, color))
#        body_counter += 1
#
#        x, y = nx, ny
#        last_dir = d
#
#    with open(spec.replace(",", "_") + ".xml", "w") as out_f:
#        print >>out_f, TEMPLATE % "\n".join(bodies)
