from setuptools import setup
import os
from glob import glob

package_name = 'testapp'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Test application for xarm7 circle motion using ROS2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'circle_motion = testapp.circle_motion:main',
            'robot_connection_test = testapp.robot_connection_test:main',
            'pick_and_place = testapp.pick_and_place:main',
            'aruco_pick_and_place = testapp.aruco_pick_and_place:main',
        ],
    },
)
