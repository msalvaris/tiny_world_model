from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='tiny_world_model',
    version='0.1.0',
    author='Mathew Salvaris',
    author_email='salvaris@example.com',
    description='A world model',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your-username/your-app-repo',
    packages=find_packages(),
    entry_points={
        'console_scripts': 
            [
                'create-dataset=world_model.simulate.ball:cli',
                'train=world_model.exp1:main',
                'generate=world_model.exp1:cli',
            ]
        },
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
)