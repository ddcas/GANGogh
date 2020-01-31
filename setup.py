from setuptools import setup, find_packages

setup(name='train',
      version='0.1',
      packages=find_packages(),
      description='train model for energy regression',
      author='Otto Nordander & Daniel Pettersson',
      author_email='daniel@soundtrackyourbrand.com',
      license='MIT',
      install_requires=[
          'tensorflow == 1.8.0',
          # 'tensorflow-gpu == 1.14.0',
          # 'keras == 2.2.0',
          # 'scikit-learn == 0.19.1',
          # 'pandas==0.23.1',
          # 'h5py == 2.8.0',
          # 'imbalanced-learn==0.3.3',
          'matplotlib==2.0.2',
          'Pillow==6.2.1',
          'imageio==2.6.1'
      ],
      include_package_data=True,
      zip_safe=False)
