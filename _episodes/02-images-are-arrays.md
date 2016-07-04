---
title: "A second part"
teaching: 5
exercises: 5
questions:
- "How should we represent images in our analysis?"

objectives:
- "Read data from neuroimaging formats using nibabel"
- "Slice and stride through data arrays with numpy indexing operations"
- "Visualize image data with Matplotlib"

keypoints:
- "Images can be efficiently and usefully represented as arrays"
- "Arrays can be manipulated with numpy operations, and visualized using Matplotlib"
---

## Image data are arrays

~~~
import nibabel as nib
img = nib.load('my_data.nii.gz')
data = img.get_data()
type(data)
~~~
{: .python}
