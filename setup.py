import subprocess
import sys

from setuptools import find_packages
from setuptools import setup


version = '1.2.5'


if sys.argv[-1] == 'release':
    commands = [
        'python setup.py sdist upload',
        'git tag v{0}'.format(version),
        'git push origin master --tag',
    ]
    for cmd in commands:
        subprocess.call(cmd, shell=True)
    sys.exit(0)


try:
    import cv2  # NOQA
except ImportError:
    print('Please install OpenCV.')
    quit(1)


install_requires = []
with open('requirements.txt') as f:
    for req in f:
        if req.startswith('-e'):
            continue
        install_requires.append(req.strip())


setup(
    name='chainer-cyclegan',
    description='Chainer Implementation of CycleGAN.',
    version=version,
    packages=find_packages(),
    install_requires=install_requires,
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    url='https://github.com/wkentaro/chainer-cyclegan',
    license='MIT',
)
