---
title: "An introduction to the image processing section"
teaching: 15
exercises: 0
questions:
- "What is image processing and how is it useful in neuroimaging?"
objectives:
- "Explain the role of image processing in neuroimaging"
- "Install and import the `scikit-image` library"
keypoints:
- Image processing is a central part of neuroimaging.
- Understanding fundamental operations in image processing is a powerful tool in building neuroimaging analysis

---

### Image processing is central to neuroimaging


### Introducing `scikit-image`

This is a Python library for image processing. It is part of the scientific
Python eco system and integrates well with `numpy`, `scipy`, `matplotlib`, and
other libraries in this ecosystem (such as `scikit-learn`). It supports both
Python 2 and Python 3.

We can use [Anaconda](http://anaconda.org) to install `scikit-image`:

~~~
conda install skimage
~~~
{: .bash}

To test that the installation worked, you can import the library as follows:
~~~
import skimage
~~~
{: .python}


> ## Other image processing libraries
>
> There are many image processing software libraries, and you might want to use
> a combination of libraries in your own applications. The following are worth
> mentioning, because they have robust open-source communities and plenty of
> documentation:
> - [OpenCV](http://opencv.org/) is a powerful image processing library with a large uptake in both industry and research and a large open-source community. The Python API is a bit cryptic, but there are multiple examples available online, that can serve as starting points for your work.
> - [ITK](https://itk.org/): the "imaging tool kit" is aimed specifically at biomedical applications. The [SimpleITK](http://www.simpleitk.org/) extension has a Python API, that can be used.
{: .callout}
