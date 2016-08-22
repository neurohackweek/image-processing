---
title: "Detrending"
teaching: 5
exercises: 5
questions:
- "How do we eliminate trends in our data"

objectives:
- "Detrend image data with a linear model"

keypoints:
- "Linear models are useful to represent the images in a concise and computationally expedient manner"

---

### Our data contains spatial bias

As we saw in the last part, MRI data often contains spatial biases. Some of
these may be due to physiological factors, such as differences in the T1
time-constant between different parts of the brain. But some of these represent
"nuisance factors" that should be eliminated as a first step in the analysis of
the image.

For example, it seems that the back of the head of this participant was closer to the measurement coil, than the front of the head. This is particularly apparent in a saggittal section.


$$ \begin{pmatrix} T1_{0,0,0} \\ T1_{0,0,1} \\ \vdots \\ T1_{0, 0, N_z} \\ T1_{0, 1, 0} \\ T1_{0,1,1} \\ \vdots \\ T1_{1,0,0} \\ T1_{1,0,1} \\ \vdots \\ T1_{N_x, N_y, N_z} \end{pmatrix} = \begin{pmatrix} x_{0,0,0} & y_{0,0,0} & z_{0,0,0} & 1 \\  x_{0,0,1} & y_{0,0,1} & z_{0,0,1} & 1\\ \vdots & \vdots & \vdots & \vdots\\ x_{0, 0, N_z} & y_{0, 0, N_z} & z_{0, 0, N_z} & 1 \\ x_{0, 1, 0} & y_{0, 1, 0} & z_{0, 1, 0} & 1\\ x_{0,1,1} & y_{0,1,1} & z_{0,1,1} & 1\\ \vdots & \vdots & \vdots & \vdots \\ x_{1,0,0} & y_{1,0,0} & z_{1,0,0} & 1\\ x_{1,0,1} & y_{1,0,1} & z_{1,0,1} & 1\\ \vdots & \vdots & \vdots & \vdots \\ x_{N_x, N_y, N_z} & y_{N_x, N_y, N_z} & z_{N_x, N_y, N_z} & 1 \end{pmatrix} $$
