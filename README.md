# Ladder Network

## Scope

Implementation of semi-supervised training technique described in Valpola et al. "Semi-Supervised Learning with Ladder Networks".
This model is used as an example to gain experience with the ```tf.estimator.Estimator``` API.

## Overview

After each layer we add normal distributed noise on each feature. The task of the decoder part is to denoise the original features of each layer.
In order to accomplish this task a connection from the distorted layer is provided to the denoising function $g(\dot, \dot)$.
The reconstruction of the intermediate features is based on the information from the upper decoder layer.