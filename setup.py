from setuptools import find_packages, setup
# to define in return typing as list of string we  need to import typing 'target type:List/Dict/..' 
from typing import List

HYPEN_E_DOT='-e .'


def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirement package that are in the file_path
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        
        # delete the "-e ." which connect the requirements.txt with setup.py from the installing the requirements package with  setup.py
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(name='mlproject',
      version='0.0.1',
      description='Machine learning application build from end to end',
      author='Yassir eljay',
      author_email='yassireljay@gmail.com',
      url='https://github.com/EljayiYassir/complet-ml-project',
      packages=find_packages(),
      install_requires=get_requirements('requirements.txt')
     )