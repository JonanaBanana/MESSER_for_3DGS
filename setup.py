from setuptools import find_packages, setup

package_name = 'messer_for_3dgs'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jonathan',
    maintainer_email='jonathan@majj.dk',
    description='A package for capturing and preparing data for gaussian splatting reconstruction without SfM from Colmap',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'isaacsim_subscriber = ros2.isaacsim_subscriber:main',
        'subscriber = ros2.subscriber:main'
        ],
    },
)
