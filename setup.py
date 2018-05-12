from setuptools import setup

setup(
     name='dicom-contour',
     packages=['dicom_contour'],    
     version=2,
     description='A library which converts DICOM images and contours into numpy arrays. An automated way of extracting image and mask voxels.',
     author='Kerem Turgutlu',
     author_email='kcturgutlu@dons.usfca.edu',
     license='MIT',
     url='https://github.com/KeremTurgutlu/dicom-contour',
     install_requires=
	['pydicom', 'numpy', 'matplotlib'],
     keywords=['dicom', 'contour', 'mask', 'medical image'],
     
)
