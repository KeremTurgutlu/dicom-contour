from setuptools import setup

setup(
     name='dicom-contour',
     packages=['dicom_contour'],    
     version=0.7,
     description='A library which converts DICOM images and contours into numpy arrays',
     author='Kerem Turgutlu',
     author_email='kcturgutlu@dons.usfca.edu',
     license='MIT',
     url='https://github.com/KeremTurgutlu/dicom-contour',
     install_requires=
	['pydicom', 'numpy', 'scipy', 'matplotlib'],
     keywords=['dicom', 'contour', 'medical image'],
     
)
