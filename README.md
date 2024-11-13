# CICM
Implementation of a cross-image co-occurrence matrix (CICM) in python, which computes a version of the common gray-level co-ocurrence matrix (GLCM) between different images, or channels of the same image.

Although python's package scikit-image implements the calculation of the GLCM matrix through method [graycomatrix](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycomatrix) in the [feature module](https://scikit-image.org/docs/stable/api/skimage.feature.html). However, since this method only accepts one image as parameter, it is not possible to compare pixel co-occurrence between pixels from different images. 

This repository provides a similar functionality as that of method graycomatrix, with the addition of a second image argument to compare pixels with. Run tests show computation times within a similar order as graycomatrix, and just like it, the computation times do not grow exponentially, but linearly, with the size of the image and the number of gray level values.

![Graph representing how computation times change, as a function of the number of gray levels, for method cicm compared with scikit-image's graycomatrix](Figure_1.png "Comparison of computation times as a function of the number of gray levels (the image size was fixed at 128 pixels)")

![Graph representing how computation times change, as a function of the image size, for method cicm compared with scikit-image's graycomatrix](Figure_2.png "Comparison of computation times as a function of the image size (number of gray levels was fixed at 256)")


