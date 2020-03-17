# keplerian_mask.py

A script to build a Keplerian mask based to be used for CLEANing or moment map analysis. This will grab the image properties (axes, beam properties and so on) from the provide CASA image.

## Basic Usage

First, load up the function into the CASA instance:

```python
CASA <X>: execfile('path/to/keplerian_mask.py')
     ...: Succesfully imported `make_mask`.
```

With this loaded, to make a Keplerian mask you just need to provide some simple geometrical properties of the disk:

```python
CASA <X>: make_mask('image_name.image',
     ...:           inc=30.0,
     ...:           PA=75.0,
     ...:           mstar=1.0,
     ...:           dist=140.0,
     ...:           vlsr=5.1e3)         
```

Hopefully the parameters are obvious, but for clarity, `inc` is the disk inclination in degrees, `PA` is the disk position angle in degrees, measured from North to the redshifted major axis in an anti-clockwise fashion. `mstar` is the stellar mass in solar masses, `dist` is the source distance in parsec and `vlsr` is the systemic velocity in meters per second.

This command will produce a new file, `image_name.mask.image`, which is the mask which can be passed to future `tclean` calls or exported as a FITS file.


## Additional Parameters

There are a few additional options to make a better fitting mask to your data.

### Inner and Outer Radii

The `r_min` and `r_max` arguments allow you to tailor the masks inner and outer radii to the emission that you observe. Both of these values are given in arcseconds.

For example, to have a ring-like mask between 0.5 and 2.5 arcseconds in (deprojected) radius:

```python
CASA <X>: make_mask('image_name.image',
     ...:           inc=30.0,
     ...:           PA=75.0,
     ...:           mstar=1.0,
     ...:           dist=140.0,
     ...:           vlsr=5.1e3,
     ...:           r_min=0.5,
     ...:           r_max=2.5)
```

### Convolution

The mask can be convolved to smooth out the edges and give a bit of a buffer between the mask edge and the emission edge. There are two ways this can be done, either by including a convolution with the rescaled beam with the `nbeams` parameter:

```python
CASA <X>: make_mask('image_name.image',
     ...:           inc=30.0,
     ...:           PA=75.0,
     ...:           mstar=1.0,
     ...:           dist=140.0,
     ...:           vlsr=5.1e3,
     ...:           nbeams=1.0)
```

Or by convolving with a circular beam with a FWHM in arcseconds given by `target_res`:

```python
CASA <X>: make_mask('image_name.image',
     ...:           inc=30.0,
     ...:           PA=75.0,
     ...:           mstar=1.0,
     ...:           dist=140.0,
     ...:           vlsr=5.1e3,
     ...:           target_res=1.0)
```

Each one of these will use CASA's `imsmooth` task to convolve the mask. As the convolution will result in non-boolean values, the `threshold` parameter dictates what is considered what is masked and what is not. A default of 0.01 is assumed, with values closer to 1 resulting in less conservative masks.

### Elevated Emission Surfaces

We can also include a non-zero emission height for molecules like 12CO. This can either by specified by a constant z/r value with the `zr` argument,

```python
CASA <X>: make_mask('image_name.image',
     ...:           inc=30.0,
     ...:           PA=75.0,
     ...:           mstar=1.0,
     ...:           dist=140.0,
     ...:           vlsr=5.1e3,
     ...:           zr=0.3)
```

If you want a more complex emission surface you can define a function which takes the midplane radius in arcseconds and returns the emission height in arcseconds.

```python
CASA <X>: def z_func(r):
     ...:     return 0.3 * r**1.5

CASA <X>: make_mask('image_name.image',
     ...:           inc=30.0,
     ...:           PA=75.0,
     ...:           mstar=1.0,
     ...:           dist=140.0,
     ...:           vlsr=5.1e3,
     ...:           z_func=z_func)
```

### Radially Varying Line Widths

With higher spatial resolutions it is possible to resolve the radially changing line width of emission lines. This manifests as a change in the width of the emission pattern as a function of radius. We assume that the radial profile is well described by a powerlaw,

![alt text](https://latex.codecogs.com/gif.latex?\Delta&space;V&space;(r)&space;=&space;\Delta&space;V_{0}&space;\times&space;\left(&space;\frac{r}{1^{\prime\prime}}&space;\right)^{\Delta&space;V_q} "Equation 1")

where `dV0` and `dVq` are parameters which can control this surface. The default values are 300 m/s for `dV0` and -0.5 for `dVq`.

```python
CASA <X>: make_mask('image_name.image',
     ...:           inc=30.0,
     ...:           PA=75.0,
     ...:           mstar=1.0,
     ...:           dist=140.0,
     ...:           vlsr=5.1e3,
     ...:           dV0=500.0,
     ...:           dVq=-0.45)
```

### Author

Written by Richard Teague (richard.d.teague@cfa.harvard.edu), 2020.
