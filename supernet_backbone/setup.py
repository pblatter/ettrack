import os
os.chdir("./lib_back")

key = 'y'
if key == 'y':
    os.system("cd apex")
    os.system("python setup.py install --cpp_ext --cuda_ext")
    os.system('cd ..')
elif key == 'n':
    pass
else:
    raise ValueError('Invalid Input')
os.system("cd ..")

