from setuptools import find_packages, setup

package_name = 'airlab_functions'

setup(
    name=package_name,
    version='0.0.0',
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
        'pc_repub = airlab_functions.pc_repub:main',
        'rgb_pcl_viz = airlab_functions.rgb_pcl_visualizer:main',
        'isaacsim_subscriber = airlab_functions.isaacsim_subscriber:main',
        'subscriber = airlab_functions.subscriber:main'
        ],
    },
)
