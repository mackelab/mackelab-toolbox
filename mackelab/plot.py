import datetime
import matplotlib.pyplot as plt

from subprocess import check_output

def saverep(basename, comment=None, pdf=True, png=True):
    """Helper function for figures

    Will create pdf and png versions of a figure and additionally generate
    a text file that contains basic info, including the hash of the latest
    git commit for reproducibility. Text passed as comment will be saved as
    well.

    Parameters
    ----------
    basename : str
        Basename for figure, without file extension
    comment : str or None (default: None)
        Comment to store in text file
    pdf : bool (default: True)
        Whether or not to save a pdf version of the figure
    png : bool (default: True)
        Whether or not to save a png version of the figure
    """
    if pdf:
        plt.savefig(basename + '.pdf')
    if png:
        plt.savefig(basename + '.png')

    info = {}
    info['basename'] = basename
    if comment is not None:
        info['comment'] = comment
    info['creation'] = datetime.datetime.now()
    try:
        info['git_revision_hash'] = check_output(['git', 'rev-parse', 'HEAD']).decode('UTF-8')[:-2]
    except:
        info['git_revision_hash'] = ''

    info_str = ''
    for key in sorted(info):
         info_str += key + ' : ' + str(info[key]) + '\n'

    with open(basename + '.txt', 'w') as textfile:
        textfile.write(info_str)
