from setuptools import setup

setup(
     name='dicom-contour',    
     version='0.1',
     description='A library which converts DICOM images and contours into numpy arrays',
     author='Kerem Turgutlu',
     author_email='kcturgutlu@dons.usfca.edu',
     license='MIT',
     install_requires=
	['dicom', 'numpy', 'scipy', 'matplotlib', 'collections', 'os', 'shutil', 'operator','warnings'],
     keywords=['dicom', 'contour', 'medical image'],
     python_requires='>=3'
)
