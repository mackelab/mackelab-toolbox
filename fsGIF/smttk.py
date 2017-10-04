import os
import re
import glob
import multiprocessing
from datetime import datetime
from parameters import ParameterSet
import sumatra.commands

import mackelab.parameters
from . import core

try:
    import click
except ImportError:
    noclick = True
else:
    noclick = False

    @click.group()
    def cli():
        pass

    @click.command()
    @click.argument('src', nargs=1)
    @click.argument('dst', nargs=1)
    @click.option('--datadir', default="")
    @click.option('--suffix', default="")
    @click.option('--ext', default=".sir")
    @click.option('--link/--no-link', default=True)
    def rename(src, dst, ext, datadir, suffix, link):
        """
        Rename a result file based on the old and new parameter files.
        TODO: Allow a list of suffixes
        TODO: Update Sumatra records database
        TODO: Match any extension; if multiple are present, ask something.
        Parameters
        ----------
        src: str
            Original parameter file
        dst: str
            New parameter file
        link: bool (default: True)
            If True, add a symbolic link from the old file name to the new.
            This avoids invalidating everything linking to the old filename.
        """
        if ext != "" and ext[0] != ".":
            ext = "." + ext
        old_params = ParameterSet(src)
        new_params = ParameterSet(dst)
        old_filename = core.RunMgr._get_filename(old_params, suffix) + ext
        new_filename = core.RunMgr._get_filename(new_params, suffix) + ext
        old_filename = os.path.join(datadir, old_filename)
        new_filename = os.path.join(datadir, new_filename)

        if not os.path.exists(old_filename):
            raise FileNotFoundError("The file '{}' is not in the current directory."
                                    .format(old_filename))
        if os.path.exists(new_filename):
            print("The target filename '{}' already exists. Skipping the"
                  "renaming of '{}'.".format(new_filename, old_filename))
        else:
            os.rename(old_filename, new_filename)
            print("Renamed {} to {}.".format(old_filename, new_filename))
        if link:
            # Allowing the link to be created even if rename failed allows to
            # rerun to add missing links
            # FIXME: Currently when rerunning the script dies on missing file
            #   before getting here
            os.symlink(os.path.basename(new_filename), old_filename)
            print("Added symbolic link from old file to new one")

    cli.add_command(rename)

    @click.command()
    @click.option('--datadir', default="data")
    @click.option('--dumpdir', default="run_dump")
    @click.option('--link/--no-link', default=True)
    @click.option('--recent/--current', default=False)
    def refile(datadir, dumpdir, link, recent):
        """
        Walk through the data directory and move files under timestamp directories
        (generated by Sumatra to differentiate calculations) to the matching
        'label-free' directory.
        E.g. if a file has the path 'data/run_dump/20170908-120245/inputs/generated_data.dat',
        it is moved to 'data/inputs/generated_data.dat'. In general,
        '[datadir]/[dumpdir]/[timestamp]/[dir1]/.../[dirn]/[filename]' -> '[datadir]/[dir1]/.../[dirn]/[filename]'

        By default, links are created from the old location to the new one. This can be
        disabled by passing the '--no-link' option.

        TODO: Get datadir from Sumatra
        TODO: Something sane when the target file already exists
        """

        lbl_pattern = '[0-9]{8}-[0-9]{6}(_[0-9]+)?'

        move_list = MoveList()
        for dirname in os.listdir(os.path.join(datadir, dumpdir)):
            if re.fullmatch(lbl_pattern, dirname) is None:
                logger.warning("Directory {} does not match the label pattern. Skipping.")
            else:
                # This is a label directory
                path = os.path.join(datadir, dumpdir, dirname)
                ## Loop over every file it contains
                for dirpath, dirnames, filenames in os.walk(path):
                    for filename in filenames:

                        ## Assemble the current path
                        old_path = os.path.join(dirpath, filename)
                        # if os.path.islink(old_path):
                        #     # Don't move symbolic links - they point to files that have already been moved
                        #     continue

                        ## Store the new (refiled) path for this file
                        split_path = old_path.split('/')
                        if split_path[0] == '':
                            # Fix for absolute paths
                            split_path[0] = '/'

                        if dumpdir == "":
                            # Since the label directory is immediately after
                            # datadir, we can get its index by seeing how many
                            # directories deep datadir is.
                            lbl_idx = len(datadir.split('/'))
                            dump_idx = None
                        else:
                            # There's a dump directory to remove as well
                            dump_idx = len(datadir.split('/'))
                            lbl_idx = dump_idx + len(dumpdir.split('/'))
                        assert(re.fullmatch(lbl_pattern, split_path[lbl_idx]))
                            # Ensure that we really are removing a timestamp directory
                        label = split_path[lbl_idx]
                        del split_path[lbl_idx]
                        if dump_idx is not None:
                            del split_path[dump_idx:lbl_idx]
                        new_path = os.path.join(*split_path)

                        ## Move the filename and create the link
                        if os.path.exists(new_path):
                            if not recent:
                                # Keep the current file version
                                print("File '{}' already exists. It was left in the labeled directory '{}'."
                                      .format(new_path, label))
                            else:
                                move_list.add_move(old_path, new_path, label)

                        else:
                            move_list.add_move(old_path, new_path, label)

        for move in move_list:
            if not os.path.islink(move['old path']):
                # Skip over links: they have already been refiled
                if os.path.exists(move['new path']):
                    renamed_path = rename_to_free_file(move['new path'])
                    print("Previous file '{}' was renamed to '{}'"
                        .format(move['new path'], renamed_path))
                else:
                    # Make sure the directory hierarchy exists
                    os.makedirs(os.path.dirname(move['new path']), exist_ok=True)

                os.rename(move['old path'], move['new path'])
                print("Refiled '{}' to the common directory."
                      .format(move['old path']))
                if link:
                    rel_new_path = os.path.relpath(move['new path'],
                                                   os.path.dirname(move['old path']))
                    os.symlink(rel_new_path, move['old path'])

    cli.add_command(refile)

    @click.command()
    @click.argument("param_file")
    @click.option("--subdir", default="")
    @click.option("--suffix", default="")
    def file_exists(param_file, subdir, suffix):
        mgr = core.RunMgr()
        if subdir == "":
            subdir = None
        if suffix == "":
            suffix = None

        searchname = mgr.get_pathname(ParameterSet(param_file), suffix, subdir)
        pathnames = glob.glob(searchname + "*")
        if len(pathnames) > 0:
            print("The following matching files were found:")
            for path in pathnames:
                print(path)
        else:
            print("No file matching '{}' was found.".format(pathname))

    cli.add_command(file_exists)

    ###########################
    # Launcher
    ###########################

    tmp_dir = "tmp"

    # TODO: Use click file arguments
    @click.command()
    @click.option("--dry-run/--run", default=False,
                  help="Use --dry to skip the computation, and just print "
                  "the command(s) that would be executed. The expanded parameter "
                  "files are left in the temporary directory to allow inspection.")
    @click.option("-n", "--cores", default=1)
    @click.option("-m", "--script", nargs=1, prompt=True)
    @click.option("--max-tasks", default=1000)
    @click.argument("args", nargs=-1)
    @click.argument("params", nargs=1)
    def run(dry_run, cores, script, max_tasks, args, params):
        basename, _ = os.path.splitext(os.path.basename(script))
        tmpparam_path = os.path.join(tmp_dir, basename + ".params")
        param_paths = mackelab.parameters.expand_param_file(
            params, tmpparam_path, max_files=max_tasks)

        # We need to generate our own label, as Sumatra's default is to use a timestamp
        # which is precise up to seconds. Thus jobs launched simultaneously would have the
        # same label. To avoid this, we generate our own label by appending a run-specific
        # number to the default time stamp label

        # Generate a timestamp label same as Sumatra's default
        timestamp = datetime.now()
        label = str(timestamp.strftime(sumatra.core.TIMESTAMP_FORMAT))
            # Same function as used in sumatra.records.Record

        argv_list = [ "-m {} --label {}_{} {} {}"
                      .format(script, label, i, " ".join(args), param_file)
                      for i, param_file in enumerate(param_paths, start=1)]
        if dry_run:
            # Dry-run
            print("With these arguments, the following calls would "
                  "distributed between {} processe{}:"
                  .format(cores, '' if cores == 1 else 's'))
            for argv in argv_list:
                print("smt run " + argv)
        else:
            if cores == 1:
                # Don't use multiprocessing. This is especially useful for debugging,
                # as execution is kept within this process
                for argv in argv_list:
                    _smtrun(argv)
            else:
                with multiprocessing.Pool(cores) as pool:
                    pool.map(_smtrun, argv_list)

    def _smtrun(argv_str):
        return sumatra.commands.run(argv_str.split())

    cli.add_command(run)

class MoveList:
    def __init__(self):
        self.moves = {}

    def __iter__(self):
        return ({'new path': key, 'old path': val['old path'],
                 'label': val['label']}
                 for key, val in self.moves.items())

    def add_move(self, old_path, new_path, label):
        if ( new_path not in self.moves
             or label > self.moves[new_path]['label']):
            self.moves[new_path] = {'old path': old_path,
                                    'label': label}

def get_free_file(path, max_files=100):
    """
    Return a file handle to an unused filename. If 'path' is free, return a handle
    to that. Otherwise, append a number to it until a free filename is found or the
    number exceeds 'max_files'. In the latter case, raise 'IOError'.

    Return a file handle, rather than just a file name, avoids the possibility of a
    race condition (a new file of the same name could be created between the time
    where one finds a free filename and then opens the file).

    Parameters
    ----------
    path: str
        Path name. Can be absolute or relative to the current directory.
    max_files: int
        (Optional) Maximum allowed number of files with the same name. If this
        number is exceeded, IOError is raised.

    Returns
    -------
    filehandle
        Filehandle, as obtained from a call to `open(pathname)`.
    pathname: str
        Pathname (including the possibly appended number) of the opened file.
    """

    # Get a full path
    # TODO: is cwd always what we want here ?
    if path[0] == '/':
        #path is already a full path name
        pathname = path
    else:
        #Make a full path from path
        pathname = os.path.abspath(path)

    # Make sure the directory exists
    os.makedirs(os.path.dirname(pathname), exist_ok=True)

    try:
        f = open(pathname, mode='xb')
        return f, pathname
    except IOError:
        name, ext = os.path.splitext(pathname)
        for i in range(2, max_files+2):
            appendedname = name + "_" + str(i) + ext
            try:
                f = open(appendedname, mode='xb')
                return f, appendedname
            except IOError:
                continue

        raise IOError("Number of files with the name '{}' has exceeded limit."
                      .format(path))

def rename_to_free_file(path):
    new_f, new_path = get_free_file(path)
    new_f.close()
    os.rename(path, new_path)
    return new_path


if __name__ == "__main__":
    if not noclick:
        cli()
