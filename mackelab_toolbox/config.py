"""
Utilities for creating project config objects

For a package called `MyPackage`, the following files should be defined:

    MyPackage/MyPackage/config.py
    MyPackage/.project-defaults.cfg

Within MyPackage/config.py, one then does something like

    from pathlib import Path
    from pydantic import BaseModel
    from mackelab_toolbox.config import ProjectConfig

    class Config(ProjectConfig):
        class PATH(BaseModel):
            <path param name 1>: <type 1>
            <path param name 2>: <type 2>
            ...
        class RUN(BaseModel):
            <run param name 1>: <type 1>
            <run param name 2>: <type 2>
            ...

    root = Path(__file__).parent.parent
    config = Config(
        path_user_config   =root/"project.cfg",
        path_default_config=root/".project-default.cfg",
        package_name       ="MyPackage")
"""
from pathlib import Path
from typing import Optional, Union, ClassVar
import logging
from configparser import ConfigParser, ExtendedInterpolation
from pydantic import BaseModel, validator
from pydantic.main import ModelMetaclass
from pydantic.utils import lenient_issubclass
import textwrap

from .utils import Singleton  # Ensure only one Config instance

logger = logging.getLogger(__name__)

# TODO: Make Config classes Singletons
class ValidatingConfigMeta(ModelMetaclass):
    """
    Some class magic with nested types:
    1. If a nested type is also used to declare a value or annotation, it is
       left untouched.
    2. If a nested type is declared but not used, do the following:
       1. Convert it to a subclass of `BaseModel` if it isn't already one.
          This allows concise definition of nested configuration blocks.
       2. Declare an annotation with this type and the same name.
          ('Type' is appended to the attribute declaring the original type,
          to prevent name conflicts.)
          Exception: If "<typename>Type" is already used in the class, and thus
          would cause conflicts, no annotation is added.
    """
    def __new__(metacls, cls, bases, namespace):
        # Use ValidatingConfig annotations as default. That way users don't need to remember to type `package_name: str = "MyProject"`
        # However, in order not to lose the default if the user *didn't* assign to that attribute,
        # we only use annotation defaults for values which are also in `namespace`.
        default_annotations = {} if cls in {"ValidatingConfig", "ValidatingConfigBase"} \
                                 else ValidatingConfig.__annotations__
        annotations = {**{nm: ann for nm, ann in default_annotations.items()
                          if nm in namespace},
                       **namespace.get("__annotations__", {})}
        if annotations:
            # Unfortunately a simple `Union[annotations.values()].__args__` does not work here
            def deref_annotations(ann):
                if isinstance(ann, Iterable):
                    for a in ann:
                        yield deref_annotations(a)
                elif hasattr(ann, "__args__"):
                    for a in ann.__args__:
                        yield deref_annotations(a)
                else:
                    yield ann
            annotation_types = set(deref_annotations(T) for T in annotations.values())
        else:
            annotation_types = set()
        attribute_types = set(type(v) for v in namespace.values())
        nested_classes = {nm: val for nm, val in namespace.items()
                          if isinstance(val, type) and nm not in {"Config", "__config__"}}
        new_namespace = {nm: val for nm, val in namespace.items()
                         if nm not in nested_classes}
        new_nested_classes = {}
        for nm, T in nested_classes.items():
            # If a declared type was used, don't touch it or its name, and don't create an associated attribute
            if T in annotation_types | attribute_types:
                new_nested_classes[nm] = T
                continue
            # Otherwise, append `Type` to the name, to free the name itself for an annotation
            # NB: This only renames the nested attribute, not the type itself
            new_nm = nm + "Type"
            if new_nm in annotations.keys() | new_namespace.keys():
                new_nm = nm  # Conflict -> no rename
            # If it isn't already a subclass of BaseModel, make it one
            if T.__bases__ == (object,):
                copied_attrs = {nm: attr for nm, attr in T.__dict__.items()
                                if nm not in {'__dict__', '__weakref__', '__qualname__', '__name__'}}
                newT = ModelMetaclass(nm, (ValidatingConfigBase,), copied_attrs)  # TODO?: Use Singleton here? (without causing metaclass conflict…)
                # newT = type(nm, (T,BaseModel), {})  
            else:
                if not issubclass(T, ValidatingConfigBase):
                    logger.warning(f"For the nested Config class '{T.__qualname__}' "
                                   "to be automatically converted to a subclass of `BaseModel`, "
                                   "it must not inherit from any other class.")
                newT = T
            new_nested_classes[new_nm] = newT
            # Add a matching annotation
            if new_nm != nm:  # Ensure we aren't overwriting the type
                annotations[nm] = newT

        return super().__new__(metacls, cls, bases,
                               {**new_namespace, **new_nested_classes,
                                '__annotations__': annotations})


class ValidatingConfigBase(BaseModel, metaclass=ValidatingConfigMeta):
    """
    Same as ValidatingConfig, without the parsing of the config file.
    Mostly used for nested entries.
    """
    # Used for passing arguments to validators -- use names that won’t cause conflicts
    make_paths_absolute: ClassVar[bool]=True

    rootdir: Path

    class Config:
        validate_all = True  # To allow specifying defaults with as little boilerplate as possible
                             # E.g. without this, we would need to write `mypath: Path=Path("the/path")`

    @validator("*", pre=True)
    def pass_rootdir(cls, val, values, field):
        "Pass the value of `rootdir` to nested subclasses"
        if lenient_issubclass(field.type_, ValidatingConfigBase):
            cur_rootdir = values.get("rootdir")
            if cur_rootdir and isinstance(val, dict) and "rootdir" not in val:
                # TODO: Any other types of arguments we should support ?
                val["rootdir"] = cur_rootdir
        return val

    # Normally it would make more sense to use pre=False; then we can just
    # check if Pydantic converted the value to a `Path`.
    # However, because "*" validators have lower precedence, any validator
    # in the Config class would get the non-prepended path.
    # By passing pre=True and doing the cast to Path ourselves, we allow
    # users to define validators which receive the absolute path.
    @validator("*", pre=True)
    def prepend_rootdir(cls, val, values, field):
        # This condition works with two types of annotations `Path` and `Union[Path, ...]`
        if Path in getattr(field.type_, "__args__", [field.type_]):
            if not isinstance(val, Path):
                val = Path(val)
            if cls.make_paths_absolute and not val.is_absolute():
                rootdir = values.get("rootdir")
                if rootdir:
                    val = rootdir/val
        return val

# TODO: Make this work with metaclass=Singleton
class ValidatingConfig(ValidatingConfigBase, metaclass=ValidatingConfigMeta):
    """
    Augments Python's ConfigParser with a dataclass interface and automatic validation.
    Pydantic is used for validation.

    The following package structure is assumed:

        code_directory
        ├── .gitignore
        ├── setup.py
        ├── project.cfg
        └── MyPkcg
            ├── [code files]
            └── config
                ├── __init__.py
                ├── .project-defaults.cfg
                └── [other config files]

    `ValidatingConfig` should be imported and instantiated from within
    ``MyPckg.config.__init__.py``.

    `project.cfg` should be excluded by `.gitignore`. This is where users can
    modify values for their local setup. If it does not exist, a template one
    is created from the contents of `.project-defaults.cfg`, along with
    instructions.

    There are some differences and magic behaviours compared to a plain
    BaseModel, which help to reduce boilerplate when defining configuration options:
    - Defaults are validated (`validate_all = True`).
    - Nested plain classes are automatically converted to inherit ValidatingConfigBase,
      and a new attribute of that class type is created. Specifically, if we
      have the following:

          class Config(ValidatingConfig):
              class paths:
                  projectdir: Path

      then this is automatically converted to

          class Config(ValidatingConfig):
              class pathsType:
                  projectdir: Path

              path : pathsType
    - A `rootdir` field is added to all auto-generated `ValidatingConfigBase`
      nested classes, and its default value is set to that of the parent.
    - All arguments of type `Path` are made absolute by prepending `rootdir`.
      Unless they already are absolute, or the class variable
      `make_paths_absolute` is set to `False`.
    """
    # rootdir: Path

    package_name: ClassVar[str]

    path_default_config: Path="config/.project-defaults.cfg"  # Rel path => prepended with rootdir
    path_user_config: Path="../project.cfg"  # Rel path => prepended with rootdir

    top_message_default: ClassVar = """
        # This configuration file is excluded from git, and so can be used to
        # configure machine-specific variables.
        # This can be used for example to set output paths for figures, or to set
        # flags (e.g. using GPU or not).
        # Default values are listed below; uncomment and edit as needed.
        #
        # Within scripts, these values are stored in the object `statGLOW.config`.
        # Adding a new config field is done by modifying the file `statGLOW/config.py`.
        
        """

    # NB: It would be nicer to use Pydantic mechanisms to deal with the defaults for
    #     path arguments. But that would require also implementing `ensure_user_config`
    #     as a validator – not sure that would actually be simpler
    def __init__(self, rootdir: Union[str,Path],
                 path_default_config=None, path_user_config=None,
                 *, make_paths_absolute: bool=True, interpolation=None,
                 **kwargs):
        """
        Instantiate a `Config` instance, reading from both the default and
        user config files.
        If the user-editable config file does not exist yet, an empty one
        with instructions is created.

        See also `ValidatingConfig.ensure_user_config_exists`.
        
        Parameters
        ----------
        rootdir: By default, all relative paths are prepended with `rootdir`.
            This is the location 
        make_paths_absolute: If true (default), all values of type 'Path'
            are prepended with `rootdir`, unless they are already absolute.
        interpolation: Passed as argument to ConfigParser. Default is
            ExtendedInterpolation(). (Note that, as with ConfigParser, an
            *instance* must be passed.)
        """
        paths = dict(path_default_config = path_default_config,
                     path_user_config = path_user_config)
        # Make paths absolute
        for nm in ("path_default_config", "path_user_config"):
            path = paths[nm] or self.__fields__[nm].default
            path = Path(path)
            if make_paths_absolute and not path.is_absolute():
                path = rootdir/path
            paths[nm] = path

        # Read the config file(s)
        if interpolation is None: interpolation = ExtendedInterpolation() 
        cfp = ConfigParser(interpolation=interpolation)
        cfp.read_file(open(paths["path_default_config"]))
        cfp.read(paths["path_user_config"])

        # Parse cfp as dict; dotted sections become nested dicts
        # TODO: Support more than one level of nesting
        cfdict = {section: dict(values) for section, values in cfp.items()}
        tomove = []
        for section in cfdict:
            if "." in section:
                section_, subsection = section.split(".", 1)
                if section_ in cfdict and subsection not in cfdict[section_]:
                    tomove.append((section_, subsection))
        for section_, subsection in tomove:
            cfdict[section_][subsection] = cfdict[f"{section_}.{subsection}"]
            del cfdict[f"{section_}.{subsection}"]

        # Create user config file if it doesn't exist yet
        # The file documents basic instructions and the default option values
        self.ensure_user_config_exists(
            paths['path_default_config'], paths['path_user_config'],
            self.package_name, **kwargs)

        # Instantiate the Config instance, using values read into `cfp`
        super().__init__(rootdir=rootdir, **paths, **cfdict)
        # if make_paths_absolute:
        #     prepend_root_to_relpaths(rootdir, self)

    def ensure_user_config_exists(
        self,
        path_default_config: Union[str,Path],
        path_user_config: Union[str,Path],
        package_name: str,
        config_module_name: str="config",
        top_message: Optional[str]=None,
        ):
        """
        If the user-editable config file does not exist, create it.

        Parameters
        ----------
        path_default_config: Path to the config file providing defaults.
            *Should* be version-controlled
        path_user_config: Path to the config file a user would modify.
            Should *not* be version-controlled
        path_config_module: Name of the python module defining config fields.
            Only used for the top message.
        package_name: The name of the package using the config object.
            Only used for the top message.
        top_message: Message to display at the top of the user config file, when it
            is created. `textwrap.dedent` is called on the value after removing
            initial newlines.
            Accepts two variables for substitution:
            `package_name` and `config_module_name`.
        """
        if top_message is None: top_message = self.top_message_default
        # Remove any initial newlines from `top_message`
        for i, c in enumerate(top_message):
            if c != "\n":
                top_message = top_message[i:]
                break
        # Finish formatting top message
        top_message = textwrap.dedent(
            top_message.format(package_name=package_name,
                               config_module_name=config_module_name))

        if not Path(path_user_config).exists():
            # The user config file does not yet exist – create it, filling with
            # commented-out values from the defaults
            with open(path_default_config, 'r') as fin:
                with open(path_user_config, 'x') as fout:
                    fout.write(textwrap.dedent(top_message))
                    stashed_lines = []  # Used to delay the printing of instructions until after comments
                    skip = True         # Used to skip copying the top message from the defaults file
                    for line in fin:
                        line = line.strip()
                        if skip:
                            if not line or line[0] == "#":
                                continue
                            else:
                                # We've found the first non-comment, non-whitespace line: stop skipping
                                skip = False
                        if not line:
                            fout.write(line+"\n")
                        elif line[0] == "[":
                            fout.write(line+"\n")
                            stashed_lines.append("# # Defaults:\n")
                        elif line[0] == "#":
                            fout.write("# "+line+"\n")
                        else:
                            for sline in stashed_lines:
                                fout.write(sline)
                            stashed_lines.clear()
                            fout.write("# "+line+"\n")
