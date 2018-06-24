"""
This script copies the matplotlib style files into the user-level config folder,
allowing them to be found by `plt.style.use`.
"""
import sys
import os
import matplotlib

# Get the target directory for matplotlib config
configdir = matplotlib.get_configdir()
os.makedirs(configdir, exist_ok=True)  # Make sure directory exists

# Get the directory containing thi module
package_root = os.path.dirname(os.path.abspath(sys.modules[__name__].__file__))

# Add directory structure to get path to `stylelib`
stylelib_path = os.path.join(package_root, 'mackelab/stylelib')

# Make sure the "stylelib" directory exists in the global config folder
os.makedirs(os.path.join(configdir, "stylelib"), exist_ok=True)

# Loop over all style files and add link in matplotlib config
for stylefile in os.listdir(stylelib_path):
    if stylefile[-9:] == '.mplstyle':
        try:
            os.symlink(os.path.join(stylelib_path, stylefile),
                       os.path.join(configdir, "stylelib", stylefile))
        except FileExistsError:
            pass
