from setuptools import setup, find_packages
import os


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
version='1.1.5'
setup(
    name="preimutils",
    packages=find_packages(),
    version=version,
    description="All you need to prepare and preprocess your annotated images",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/mrl-amrl/preimutils",
    download_url='https://github.com/mrl-amrl/preimutils/archive/{}.tar.gz'.format(version),
    author="Amir Sharifi",
    author_email="ami_rsh@outlook.com",
    license='MIT',
    keywords=['computer vision', 'image processing', 'opencv',
              'matplotlib', 'preprocess image', 'image dataset', 'pascal voc'],
    entry_points={
        'console_scripts': ['preimutils=preimutils.object_detection.run:main'],
    },
    install_requires=[
        'pascal_voc_writer',
        'tqdm',
        'albumentations',
        'pandas',
        'imutils',
        'opencv-python',
        'imageio',
        'pycocotools',
        'shutils',
        'xmltodict',
        'numpy',
        'matplotlib',
    ],
    zip_safe=False,
)
