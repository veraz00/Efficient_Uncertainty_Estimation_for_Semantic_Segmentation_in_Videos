import sys 
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.join(this_dir, '..')
if main_dir not in sys.path:
    sys.path.insert(0, main_dir)
print('main_dir', main_dir)