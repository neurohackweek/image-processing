---
title: "Introduction: what is image processing"
teaching: 15
exercises: 1
questions:
- "What is image processing and how is it useful in neuroimaging?"
- "How are images represented in scientific computing?"

objectives:
- "Explain the role of image processing in neuroimaging"
- "Identify image data, and distinguish it from other data (e.g., tabular, time-series, etc.)"
- "Extract image data from neuroimaging file formats with nibabel"
- "Slice and stride through data arrays with numpy indexing operations"
- "Visualize image data with Matplotlib"

keypoints:
- "Image processing operatins are a central part of neuroimaging."
- "Images are homogenous arrays, where spatial relationships are important"
- "Many different operations can be performed on images, and processing pipelines can be build combining them"
- "Images can be efficiently and usefully represented as arrays"
- "Arrays can be manipulated with numpy operations, and visualized using Matplotlib"
---

### Image processing is central to neuroimaging

Image processing is a large and very general set of tools that are used across a
variety of research disciplines to analyze image data. Naturally, image
processing algorithms are fundamental to neuroimaging, because a lot (if not
all) the data that we analyze in neuroimaging is image data.

What is image data? How is it different from other data, such as time-series,
or tabular data?

For our purposes: image data is defined as multi-dimensional homogenous data in
which *spatial relationships matter*. That is, neighboring pixels are treated
differently than pixels from disparate parts of the array. Spatial contiguity is
meaningful. Usually, image data will have 2 or 3 dimensions, corresponding to
the 3 spatial dimensions or 2D projection: either from a specific view-point
(think photographs) or through a 3D object (think slice). But it is possible to
use image processing algorithms in cases in which there are more dimensions, and
where the dimensions do not correspond to the spatial dimensions (does anyone
know a good example of this?)

Note that these categories are also not mutually exclusive. For example,
functional MRI data is image data, but is also time-series data at the same
time.

#### Some common image processing operations

There are many different kinds of image processing operations. Here are a few common operations:


- Filtering
  - Detrending
  - Denoising
  - Smoothing
- Segmentation
- Feature detection
- Texture analysis
- Statistical characterization
- Classification
- Registration
- Combination (e.g. 'stitching')


### Images can be represented in arrays

Because of their nature (homogenous/spatial dimensions matter) data lend
themselves easily to a representation as arrays. Let's demonstrate this with
some data from the Human Connectome Project.

> ## Downloading data from the Human Connectome Project
>
> The [Human Connectome Project](https://www.humanconnectome.org/) provides
> high-quality functional, structural and diffusion MRI data. These can be
> accessed through [AWS Simple Storage Service](https://aws.amazon.com/s3/),
> or "S3". This allows us to programatically downlaod the data, through a
> Python library called [`boto3`](https://github.com/boto/boto3).
> For the following code to work, you need to have a file '~/.aws/credentials',
> that includes a section:
>
>   [hcp]
>   AWS_ACCESS_KEY_ID=XXXXXXXXXXXXXXXX
>  AWS_SECRET_ACCESS_KEY=XXXXXXXXXXXXXXXX
>
> The keys are credentials that you can get from [HCP] (https://wiki.humanconnectome.org/display/PublicData/How+To+Connect+to+Connectome+Data+via+AWS)
>
> In addition, you'll need to install `boto3`. This can usually be done with the following command-line call:
>
>     pip install boto3

First, we download the data to our computer from S3 using `boto3`.

~~~
import boto3
boto3.setup_default_session(profile_name='hcp')
s3 = boto3.resource('s3')
bucket = s3.Bucket('hcp-openaccess')
subject = 991267  # We can replace that with other subject IDs!
bucket.download_file('HCP/%s/T1w/T1w_acpc_dc.nii.gz'%subject,
                     '%s-T1w_acpc_dc.nii.gz'%subject)
~~~
{: .python}

After running this code, we should have T1-weighted MRI scan of this subject
stored in the file `991267-T1w_acpc_dc.nii.gz`. We can read this file into
memory using the `nibabel` library.

> ## Nibabel: harnesssing the cacophony of neuroimaging file
>
> One of the challenges of data science in neuroimaging (and in other
> scientific fields) is the range of different file formats that are used to
> store data. Often these files will be opaque to a naive user, because the data
> is stored in a binary format, that cannot be read without knowledge of the
> organization of the data on disks.
>
> The `nibabel` library alleviates these difficulty through a careful
> implemntation of a wider range of different neuroimaging file-formats.
> Wherever possible, the library presents a common interface to these different
> file formats, making it particularly easy to write code that will work on
> data stored in these different formats.
>
> To install it, you can use the following command-line call:
>
>     pip install nibabel

The `nibabel` API for reading data from file has two steps:

~~~
import nibabel as nib
T1w_img = nib.load('991267-T1w_acpc_dc.nii.gz')
~~~
{: .python}

Because `nibabel` loads the data "lazily", the data hasn't been read into memory
yet, only some basic metadata stored in the file header. To access the data, we
need to explicitly call the `get_data` method of the image object that we
currently have in memory:

~~~
T1w_data = T1w_img.get_data()
~~~
{: .python}

The data is stored in a `numpy` array. We can verify that by running:

~~~
type(T1w_data)
~~~
{: .python}

We can check some basic properties of this array by running:

~~~
T1w_data.shape
T1w_data.dtype
~~~
{: .python}

We can also visualize the data that was stored in the file using `Matplotlib`

~~~
import matplotlib.pyplot as plt
%matplotlib inline

fig, ax = plt.subplots(1)
ax.matshow(T1w_data[:, :, T1w_data.shape[-1]//2])
~~~
{: .python}

Exercise
