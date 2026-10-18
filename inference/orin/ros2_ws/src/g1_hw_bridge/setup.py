from setuptools import find_packages, setup

package_name = 'g1_hw_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mimi',
    maintainer_email='mimi@example.com',
    description='Hardware bridge boundary for Unitree G1 (mock + real).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mock_hw_bridge_node = g1_hw_bridge.mock_hw_bridge_node:main',
            'real_hw_bridge_node = g1_hw_bridge.real_hw_bridge_node:main',
        ],
    },
)
