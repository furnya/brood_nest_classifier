#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lines = (line.strip() for line in open(filename))
    return [line for line in lines if line and not line.startswith("#")]


reqs = parse_requirements("requirements.txt")
dep_links = [url for url in reqs if "http" in url]
reqs = [req for req in reqs if "http" not in req]
reqs += [url.split("egg=")[-1] for url in dep_links if "egg=" in url]


setup(
    name='bn_classifier',
    version='0.1',
    description='Brood nest classifier',
    author='Lino Steinhau',
    author_email='lino.steinhau@fu-berlin.de',
    # url='https://github.com/BioroboticsLab/bb_utils/',
    install_requires=reqs,
    dependency_links=dep_links,
    include_package_data=True,
    packages=['bn_classifier'],
    package_dir={'bn_classifier': 'bn_classifier'},
    package_data={'bn_classifier': [
        'weights/best-augment.pth'
        # 'data/hatchdates2016.csv',
        # 'data/foragergroups2016.csv',
        # 'data/beenames.csv',
        # 'data/fiducial_marker.npz',
        # 'data/idmapping2019.csv'
    ]},
    entry_points={
        'console_scripts': [
            # 'bb_gt_to_hdf5 = bb_utils.scripts.gt_to_hdf5:run',
            # 'shuffle_hdf5 = bb_utils.scripts.shuffle_hdf5:main',
        ]
    },
    scripts=[
        # 'scripts/shuffle_all_hdf5.sh',
        # 'scripts/build_gt.sh',
    ]
)
