---
title: "Detrending"
teaching: 5
exercises: 5
questions:
- "How do we eliminate trends in our data"

objectives:
- "Detrend image data with a linear model"
-

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
