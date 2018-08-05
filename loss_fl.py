# Copyright 2018 BIG CHENG (bigcheng.asus@gmail.com). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

## BIG CHENG, 2018/08/01, init, loss for facial landmark

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math

_NUM_FACE_LANDMARKS = 68

class loss_fl():
  num_lms = _NUM_FACE_LANDMARKS

  ## flms, ground truth tensor for facial landmark, typically (32, 136)
  ## flms, prediction value tensor for facial landmark, typically (32, 136)
  ## w_lms, weight for landmark
  ## w_curve, weight for "curve" constraint
  ## (todo) w_nonzero
  #def __init__(self, flms, logits, w_lms, w_curve):
  def __init__(self, flms, logits):
    self.flms = flms
    self.logits = logits

    #self.w_lms = w_lms
    #self.w_curve = w_curve
    

  #def add_m2e(logits, flms, w_lms=1):  ## use w_lms to high light some landmark
  def add_lms_err(self, w_lms=1., w_eye=1.):  ## use w_lms to high light some landmark
    w_base = 1.
    #w_middle = math.sqrt(w_lms*w_base)
    #w_middle = w_eye
    weights_i = [w_base] * 68
    #print "weights_i= ", weights_i
    ## 1, 9, 17 for face highlight
    weights_i[0] = w_lms 
    weights_i[8] = w_lms  
    weights_i[16] = w_lms  
    ## 20, 25 for eyebow highlight
    weights_i[19] = w_lms  
    weights_i[24] = w_lms
    ## 34 for nose highlight
    weights_i[33] = w_lms  
    ## 40, 43 for eyes highlight
    weights_i[36:42] = [w_eye] * 6
    weights_i[39] = max(w_lms, w_eye)  
    weights_i[42:48] = [w_eye] * 6
    weights_i[42] = max(w_lms, w_eye)  
    ## 49, 55 for mouse highlight
    weights_i[48] = w_lms  
    weights_i[54] = w_lms  

    #pts_face = range(1, 17)
    #pts_rbow = range(18, 22)
    #pts_lbow = range(23, 27)
    #pts_nose = range(28, 31)
    #pts_bnose = range(32, 36)
    #pts_reye = range(37, 42)
    #pts_leye = range(43, 48)
    #pts_mouse = range(49, 68)

    print "weights_i= ", weights_i

    weights_v = tf.constant(value=(weights_i+weights_i), shape=(1, loss_fl.num_lms*2))
    print "weights_v= ", weights_v

    #loss = tf.losses.mean_squared_error(logits, flms)
    print "self.logits", self.logits
    print "self.flms", self.flms
    loss = tf.losses.mean_squared_error(self.logits, self.flms, weights = weights_v)
    print "loss= ", loss  #Tensor("mean_squared_error/value:0", shape=(), dtype=float32, device=/device:CPU:0)

    return loss


  ## contraint      

  """
  def add_2pt_lost(idx1, idx2, weights):  ## 1-based index
    pt1 = get_pt(idx1) 
    pt2 = get_pt(idx2) 
    #print "pt1= ", pt1  #pt1=  Tensor("Slice:0", shape=(32, 1), dtype=float32, device=/device:CPU:0)
    #print "pt2= ", pt2  #pt2=  Tensor("Slice:0", shape=(32, 1), dtype=float32, device=/device:CPU:0)
    tf.losses.mean_squared_error(pt1, pt2, weights=weights)
  """

  def add_curve_err(self, w_curve=1e-3):  ## typical gt = 4-5k

    def get_pt(idx): ## 1-based index
      pt_x = self.logits[:, idx-1:idx]
      pt_y = self.logits[:, idx-1+loss_fl.num_lms:idx+loss_fl.num_lms]
      return tf.concat((pt_x, pt_y), axis=1)

    def add_pt_lost(idx, weights): ## 1-based index
      #pt_x = logits[:, idx1-1:idx1];
      #pt_y = logits[:, idx1-1+_NUM_FACE_LANDMARKS:idx1+_NUM_FACE_LANDMARKS];
      pt1 = get_pt(idx) 
      pt2 = get_pt(idx+1) 
      #print "pt1= ", pt1  #pt1=  Tensor("Slice:0", shape=(32, 1), dtype=float32, device=/device:CPU:0)
      #print "pt2= ", pt2  #pt2=  Tensor("Slice:0", shape=(32, 1), dtype=float32, device=/device:CPU:0)
      print "pt %d weight= " % idx, weights 
      tf.losses.mean_squared_error(pt1, pt2, weights=weights)

    def add_pts_lost(idxes, weights):
      for idx in idxes:
        add_pt_lost(idx, weights)

    pts_face = range(1, 17)
    pts_rbow = range(18, 22)
    pts_lbow = range(23, 27)
    pts_nose = range(28, 31)
    pts_bnose = range(32, 36)
    pts_reye = range(37, 42)
    pts_leye = range(43, 48)
    pts_mouse = range(49, 68)
    #pts_all = pts_face + pts_rbow + pts_lbow + pts_nose + pts_bnose + pts_reye + pts_leye + pts_mouse

    add_pts_lost(pts_face, w_curve)  ## 
    add_pts_lost(pts_rbow, w_curve)  ## 
    add_pts_lost(pts_lbow, w_curve)  ## 
    add_pts_lost(pts_nose, w_curve)  ## 
    add_pts_lost(pts_bnose, w_curve)  ## 
    add_pts_lost(pts_reye, w_curve)  ## 
    add_pts_lost(pts_leye, w_curve)  ## 
    add_pts_lost(pts_mouse, w_curve)  ## 

    #add_2pt_lost(42, 37, 0.00001)  ## eye
    #add_2pt_lost(48, 43, 0.00001)  ## eye
    #add_2pt_lost(31, 34, 0.00001)  ## nose
    #add_2pt_lost(60, 49, 0.00001)  ## mouse
    #add_2pt_lost(68, 61, 0.00001)  ## mouse

    #return None


  def add_nonzero_err(self, w_nz=1e4):
    #shape_logits = tf.shape(self.logits)
    #print shape_logits, tuple(shape_logits)
    #target_v = tf.constant(value=2, shape=tuple(shape_logits))  ##  2/224 = 2%
    target_v = tf.fill(tf.shape(self.logits), 1.0)

    #wc = 10000000*32
    #tf.losses.mean_squared_error(logits, center_v, weights = wc)
    tf.losses.hinge_loss(target_v, self.logits, weights = w_nz)  ## average by 32


if __name__ == "__main__":
  
  def utest1():

    # Launch the default graph.
    with tf.Session() as sess:

      logits = tf.constant(value=[1, 1, 5, 4], shape=(32,_NUM_FACE_LANDMARKS*2))
      flms = tf.constant(value=[1, 1, 3, 4], shape=(32,_NUM_FACE_LANDMARKS*2))

      loss = tf.losses.mean_squared_error(logits, flms, weights = 1)
      #loss = add_m2e(logits, flms)

      print("logits", sess.run(logits))
      print("flms", sess.run(flms))
      print("loss", sess.run(loss))
      
      #add_pts_err()


  def utest2():
    # Launch the default graph.
    with tf.Session() as sess:

      logits = tf.constant(value=[1.0, 0.5], shape=(3,2))
      flms = tf.constant(value=[1, 1], shape=(3,2))

      loss = tf.losses.hinge_loss(logits, flms, weights = 1)

      print("logits", sess.run(logits))
      print("flms", sess.run(flms))
      print("loss", sess.run(loss))

  ## test lms err
  def utest10():

    # Launch the default graph.
    with tf.Session() as sess:

      logits = tf.constant(value=[1.01, 1.0], shape=(32, _NUM_FACE_LANDMARKS*2))
      flms = tf.constant(value=[1, 1], shape=(32, _NUM_FACE_LANDMARKS*2))

      loss = loss_fl(flms, logits).add_lms_err(10)

      print("logits", sess.run(logits))
      print("flms", sess.run(flms))
      print("loss", sess.run(loss))


  ## test curve err
  def utest11():

    # Launch the default graph.
    with tf.Session() as sess:

      #logits = tf.constant(value=[1., 1.0], shape=(32, _NUM_FACE_LANDMARKS*2))   ## loss = 0
      #logits = tf.constant(value=[1., 9.0], shape=(32, _NUM_FACE_LANDMARKS*2))   ## (1-9)**2/2 * (1/32) * 10 (weight) = 10
      logits = tf.constant(value=range(_NUM_FACE_LANDMARKS*2), shape=(1, _NUM_FACE_LANDMARKS*2))   ## (2)**2/2(*x-y-x)/2(*2points) * (68-8) * 10 (weight) = 600
      #flms = tf.constant(value=[1, 1], shape=(32, _NUM_FACE_LANDMARKS*2))

      loss_fl(None, logits).add_curve_err(10)
      loss = tf.losses.get_total_loss()

      print("logits", sess.run(logits))
      #print("flms", sess.run(flms))
      print("loss", sess.run(loss))


  ## test none-zero err
  def utest12():

    # Launch the default graph.
    with tf.Session() as sess:

      #logits = tf.constant(value=2, shape=(1, _NUM_FACE_LANDMARKS*2))     ## 0
      #logits = tf.constant(value=0, shape=(1, _NUM_FACE_LANDMARKS*2))  ## 1e4 
      logits = tf.constant(value=[0,1,1,2], shape=(1, 4))  ## 2500 

      loss_fl(None, logits).add_nonzero_err(1e4)
      loss = tf.losses.get_total_loss()

      print("logits", sess.run(logits))
      #print("flms", sess.run(flms))
      print("loss", sess.run(loss))


  utest12()

