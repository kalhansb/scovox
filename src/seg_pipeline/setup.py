from setuptools import setup

package_name = 'seg_pipeline'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kalhansb',
    maintainer_email='kalhansandaru@gmail.com',
    description='Online RGB-D semantic segmentation node for SCovox.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'seg_node = seg_pipeline.seg_node:main',
        ],
    },
)
