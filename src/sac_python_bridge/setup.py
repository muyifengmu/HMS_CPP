from setuptools import find_packages, setup

package_name = 'sac_python_bridge'

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
    maintainer='cmh',
    maintainer_email='muyifengmu@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sac_server_grid = sac_python_bridge.sac_server_grid:main',
            'automated_training = sac_python_bridge.automated_training:main',
            'p3d_tf_broadcaster = sac_python_bridge.p3d_tf_broadcaster:main'
        ],
    },
)
