from setuptools import setup

setup(
    name="preimutils",
    packages=['preimutils'],
    version="0.0.1",
    description="all you need to prepare and preprocess your annotated images",
    url="https://github.com/mrl-amrl/preimutils",
    download_url='https://github.com/mrl-amrl/preimutils/archive/0.0.1.tar.gz',
    author="Amir Sharifi",
    author_email="ami_rsh@outlook.com",
    license='MIT',
    install_requires=[
        'pascal_voc_writer',
        'tqdm',
        'pandas'
    ],
    zip_safe=False,       
)
