from theano import function, config, shared, tensor
import numpy
import logging
logger = logging.getLogger(__name__)

def using_gpu():
    """
    Return True if Theano is currently able to using the GPU. This function
    involves compiling a small function, and so takes a few seconds to execute.
    """
    # Based on a test script found here: http://deeplearning.net/software/theano/tutorial/using_gpu.html#testing-theano-with-gpu
    x = shared(numpy.arange(8))
    f = function([], tensor.exp(x))
    if numpy.any([isinstance(x.op, tensor.Elemwise) and
                ('Gpu' not in type(x.op).__name__)
                for x in f.maker.fgraph.toposort()]):
        return False
    else:
        return True

# =====================================
# Caching graphs and compiled graphs
# =====================================
# TODO: Don't repeat code between GraphCache and CompiledGraphCache
# TODO: Key CompiledGraphCache on all function args (input, output, flags…)

import shelve
import builtins
import inspect
from collections import Counter, OrderedDict
import theano_shim as shim
from . import utils
class CachingError(RuntimeError):
    # TODO: Check `on_cache_fail` here, rather than in `GraphCache.set`
    pass
class GraphCache:
    """
    A cache which stores computational graphs on disk and which will flush
    itself whenever dependent code changes.
    This prevents that we retrieve the graphs constructed with old code.
    This class is used for caching graph transformations: both keys and values
    are graphs, respectively corresponding to the pre- and post-transformation
    graphs.
    More precisely: values are `(graph, updates)` tuples, where `updates` is
    an update dictionary.

    ..Note The behaviour when encountering errors can be adjusted by setting
    `GraphCache.on_cache_fail` (to modify all caches) or `cache.on_cache_fail`
    (to modify a specific cache `cache`). Possible values are:
        - 'ignore': Do nothing.
        - 'warn'  : Print a message with `logger.error`.
        - 'raise' : Raise a CachingError exception.
    The default setting of 'warn' ensures that failing to cache a graph does
    not prevent execution of otherwise functioning code.

    Parameters
    ----------
    cachename: str
        On-disk filename of the cache.
    *Classes: Types
        Class types. If any of these change, including types they inherit from,
        the on-disk cache will be cleared.
    modules: list of modules (keyword-only)
        List of modules which when changed, should invalidate a cache.
        If a module is imported as `import foo`, then pass it as
        `GraphCache([cachename], [classes], foo)`.
    """
    on_cache_fail = 'warn'  # One of 'ignore', 'warn', 'raise'
    num_dep_caches = 5  # Keep caches for this many different dependency hashes
                        # Least recently used dependencies are dropped when
                        # their number exceeds `num_dep_caches`

    def __init__(self, cachename, *Classes, modules=()):
        self.cachename = cachename

        if type(self) not in Classes:
            Classes += (type(self),)
        # Compute a hash based on all the class dependencies
        classhashes = []
        for cls in Classes:
            if not isinstance(cls, type):
                raise TypeError("`Classes` arguments to `GraphCache` "
                                " initializer must be types (i.e. classes).")
            for C in inspect.getmro(cls):
                try:
                    src = inspect.getsource(C)
                except TypeError:
                    # Can't get source for built-in types, or which are
                    # not defined in a source file (e.g. those defined in a
                    # REPL or a Jupyter notebook).
                    # The builtin types we don't care since they shouldn't
                    # change, but for the others we should at least warn the
                    # user.
                    if getattr(builtins, C.__name__, None) is C:
                        # Built-in type, no worries
                        pass
                    else:
                        # Non built-in type, print warning
                        logger.warning(
                            "GraphCache '{}' will ignore dependencies on type "
                            "'{}', because it was not defined in a source "
                            "file. This can happen e.g. if you try to create "
                            "a cache depending on a class defined within a "
                            "Jupyter notebook."
                            .format(cachename, C))
                else:
                    # We got the class' source code, now hash it
                    classhashes.append(utils.stablehash(src))
        classhashes = tuple(classhashes)
        # Add the module dependencies
        if not isinstance(modules, (list, tuple)):
            modules = (modules,)
        modulehashes = tuple(utils.stablehash(inspect.getsource(m))
                             for m in modules)
        dependencyhash = utils.stablehash(classhashes+modulehashes)

        # Open the cache (`shelve` will create it if needed) and check
        # if the dependency hash is new. If it is and the we already have
        # the maximum number of caches, delete the oldest.
        # Dependency hashes are used as keys: for each dependency, we store
        # a dictionary of cached objects. Hashes are also stored in the
        # 'dependencyhashes' entry in a list. Every time we use a hash,
        # it is moved to the end of the list, such that the first element
        # is always the least recently used.
        with shelve.open(self.cachename) as cache:
            if 'dependencyhashes' not in cache:
                cache['dependencyhashes'] = []
            dephashes = cache['dependencyhashes']

            if dependencyhash in cache['dependencyhashes']:
                dephashes.remove(dependencyhash)
                    # Remove from list; we'll add it back to the end
            elif len(dephashes) > self.num_dep_caches:
                logger.debug("Maximum number of caches exceeded; clearing "
                             "oldest.")
                oldest = dephashes[0]
                if oldest in cache:
                    del cache[oldest]  # Remove caches for this dependency
                del dephashes[0]   # Remove dependency from dep. list
            dephashes.append(dependencyhash)
            # All dependency hashes should be unique
            assert len(dephashes) == len(set(dephashes))
            cache['dependencyhashes'] = dephashes
            cache[dependencyhash] = {}

    def __str__(self):
        with shelve.open(self.cachename) as cache:
            d = dict(cache)
        return str(d)

    def __setitem__(self, key_graph, value):
        return self.set(key_graph, value)

    @staticmethod
    def get_graph_hash(graph):
        return str(shim.graph.hash(graph))

    @staticmethod
    def associate_by_attr(targets, sources, match_attr):
        """
        Return a dictionary of `{target: source}` pairs which are matched based
        on their value of :param:match_attr.
        Raises an error if either :param:targets or :param:sources do not all
        have the attribute :param:match_attr, or if some of the values for
        that attribute are duplicated.
        """
        # Standardize inputs
        if isinstance(targets, shim.cf.TerminatingTypes):
            targets = [targets]
        elif not isinstance(targets, (list, tuple)):
            # Make sure we have an indexable sequence
            targets = list(targets)
        if isinstance(sources, shim.cf.TerminatingTypes):
            sources = [sources]
        elif not isinstance(sources, (list, tuple)):
            # Make sure we have an indexable sequence
            sources = list(sources)
        # Test that both targets and sources all have unique attributes
        if not all(hasattr(t, match_attr) for t in targets):
            raise AttributeError("Some of the elements in `targets` do not "
                                 "have the attribute '{}'".format(match_attr))
        if not all(hasattr(s, match_attr) for s in sources):
            raise AttributeError("Some of the elements in `sources` do not "
                                 "have the attribute '{}'".format(match_attr))
        target_attrs = [getattr(t, match_attr) for t in targets]
        source_attrs = [getattr(s, match_attr) for s in sources]
        counts = Counter([getattr(v, match_attr) for v in targets])
        dup_attrs = [attr for attr, count in counts.items() if count > 1]
        if len(dup_attrs) > 0:
            raise AttributeError("Some targets have "
                                 "duplicated names: {}".format(dup_attrs))
        counts = Counter([getattr(v, match_attr) for v in sources])
        dup_attrs = [attr for attr, count in counts.items() if count > 1]
        if len(dup_attrs) > 0:
            raise AttributeError("Some sources have "
                                 "duplicated names: {}".format(dup_attrs))

        # Build the substitution dictionary
        subdict = {}
        for t, attr in zip(targets, target_attrs):
            try:
                i = source_attrs.index(attr)
            except ValueError:
                raise AttributeError(
                    "None of the source variables has an attribute '{}' "
                    "matching '{}'.".format(match_attr, attr))
            subdict[t] = sources[i]

        return subdict

    @staticmethod
    def associate_by_name(targets, sources):
        return GraphCache.associate_by_attr(targets, sources, 'name')

    def get(self, graph, updates=None, other_inputs=(), rng=None):
        """
        Cache key is a combination of :param:graph and :param:updates.
        Since the cached graphs are typically manipulated versions of the keys,
        they may involve symbolic inputs which are not inputs to the keys.
        Matching variables for these extra inputs need to be provided with
        :param:other_inputs.

        ..FIXME: Currently does not work if the RNG depends on a shared
        variable. So instead of doing `rng.normal(std=σ)`, do
        `σ*rng.normal()`.
        Possible solution: completely replace the loaded RNG with the current
        one.

        Parameters
        ----------
        graph: Symbolic expression | list of symbolic expressions
            Symbolic expression which served as a key for the cache.
        updates: OrderedDict
            Update dictionary. Also served as a key for the cache.
        other_inputs: list of symbolic variables
            List of other symbolic expressions which may appear in the graph
            loaded from cache.
        rng: random stream object (shim.RandomStreams or symbolic equivalent)
            If graphs depend on random numbers, the source RNG must be provided.
        """
        graphs = [graph] if isinstance(graph, shim.cf.GraphTypes) else graph
        if updates is None: updates = OrderedDict()
        all_graphs = (graphs + list(updates.keys())
                      + list(updates.values()) + list(other_inputs))
        inputs = shim.graph.symbolic_inputs(all_graphs)
        with shelve.open(self.cachename) as cache:
            cache = cache[cache['dependencyhashes'][-1]]
                # Latest dependency hash is always the current one
            graphhash = self.get_graph_hash(graph)
            updateshash = self.get_graph_hash(updates.items())
            r = cache.get(graphhash+updateshash, None)
        if r is None:
            logger.debug("GraphCache miss")
            return None, None  # Expected format is `graph, updates`
        else:
            logger.debug("GraphCache hit")
            load_graph, load_updates, load_rng = r
            # The graph loaded from cache will have created all new variables
            # We need to substitute the current ones, based on them having the
            # same name.

            # Copy the state of the random number generator
            if (rng is None) and (load_rng is not None):
                msg = ("Cannot load graph from cache: cached graph "
                       "requires a random number generator")
                if self.on_cache_fail == 'warn':
                    logger.error(msg)
                elif self.on_cache_fail == 'raise':
                    raise CachingError(msg)
            if load_rng is not None:
                # TODO: include symbolic substitutions in `copy_random_state`
                # `rng.state_updates` returns a list of tuples of the form
                # `(state update, random function)`
                # The former is a shared variable which is updated in
                # `copy_random_state`, the latter a symbolic expression
                rng_upds = [u[1] for u in rng.state_updates]
                loadrng_upds = [u[1] for u in load_rng.state_updates]
                rng_inputs = [
                    v for v in shim.graph.symbolic_inputs(rng_upds)
                      if not isinstance(v, shim.cf.ConstantTypes)
                         and not isinstance(v, shim.cf.RandomStateType)]
                loadrng_inputs = [
                    v for v in shim.graph.symbolic_inputs(loadrng_upds)
                      if not isinstance(v, shim.cf.RandomStateType)
                         and not isinstance(v, shim.cf.ConstantTypes)]
                subdict = self.associate_by_name(loadrng_inputs, rng_inputs)
                for i in range(len(load_rng.state_updates)):
                    su = load_rng.state_updates[i]
                    load_rng.state_updates[i] = \
                        (su[0], shim.graph.clone(su[1], replace=subdict))
                shim.copy_random_state(from_rng=rng, to_rng=load_rng)

            # Remove variables from inputs which have no name
            inputs = [v for v in inputs
                        if hasattr(v, 'name') and v.name is not None]
            # Build the substitution dictionary
            assert type(graph) is type(load_graph)
            if not isinstance(graph, (list, tuple)):
                load_graphlist = [load_graph]
                return_type = shim.cf.SymbolicType
            else:
                load_graphlist = load_graph
                return_type = type(graph)
            vars_to_sub = shim.graph.symbolic_inputs(
                load_graphlist + list(load_updates.keys()) +
                list(load_updates.values()))
            vars_to_sub = [v for v in vars_to_sub
                             if not isinstance(v, shim.cf.RandomStateType)]
                # Random state substitution taken care of separately
            try:
                subdict = self.associate_by_name(vars_to_sub, inputs)
            except AttributeError as e:
                msg = "Cannot load graph from cache: " + e.args[0]
                if self.on_cache_fail == 'warn':
                    logger.error(msg)
                elif self.on_cache_fail == 'raise':
                    raise CachingError(msg)
                return None, None  # Abort

            # Perform the substitutions
            graphs = [shim.graph.clone(g, subdict) for g in load_graphlist]
            updates = OrderedDict((shim.graph.clone(key, subdict),
                                   shim.graph.clone(val, subdict))
                                  for key, val in load_updates.items())
            assert not any(
                v in vars_to_sub
                for v in shim.graph.symbolic_inputs(
                    graphs + list(updates.keys()) + list(updates.values())))

            # Return the graph and updates
            if isinstance(return_type, tuple):
                graphs = tuple(graphs)
            elif isinstance(return_type, shim.cf.SymbolicType):
                assert len(graphs) == 1
                graphs = graphs[0]
            return graphs, updates

    def set(self, key_graph, key_updates, val_graph=None, val_updates=None,
            rng=None):
        """
        TODO: update following: Accepts both :param:val_graph, :param:updates and :param:rng separately
        or as a tuple.
        :param:val_graph itself may be a single symbolic expression, or a
        list or tuple of such expressions.
        """
        # Check that argument types are correct
        if val_graph is None:
            # key, val are either packaged as tuples, or consist of only
            # a graph.
            assert val_updates is None
            assert rng is None
            val_graph = key_updates
            if (isinstance(key_graph, shim.cf.GraphTypes)
                or (isinstance(key_graph, (list, tuple)) and
                    all(isinstance(g, shim.cf.GraphTypes) for g in key_graph))):
                key_updates = OrderedDict()
            else:
                assert isinstance(key_graph, tuple)
                if len(key_graph) == 1:
                    key_graph = key_graph[0]
                    key_updates = OrderedDict()
                elif len(key_graph) == 2:
                    key_graph, key_updates = key_graph
                else:
                    raise TypeError("If using a tuple for a cache key, it "
                                    "must be of the form `(graph, updates)`.")
            if (isinstance(val_graph, shim.cf.GraphTypes)
                or (isinstance(val_graph, (list, tuple)) and
                    all(isinstance(g, shim.cf.GraphTypes) for g in val_graph))):
                val_updates = OrderedDict()
            else:
                assert isinstance(key_updates, tuple)
                if len(val_graph) == 1:
                    val_graph = val_graph[0]
                    val_updates = None
                elif len(val_graph) == 2:
                    val_graph, val_updates = val_graph
                elif len(val_graph) == 3:
                    val_graph, val_updates, rng = val_graph
                else:
                    raise TypeError(
                        "If setting a cached value with a tuple, it must be "
                        "of the form `(graph, updates)` or `(graph, updates, "
                        "rng)`.")

        # if not isinstance(key_graph, shim.cf.SymbolicType):
        #     raise TypeError(
        #         "`key_graph` must be a symbolic expression.")
        # if not isinstance(val_graph, shim.cf.SymbolicType):
        #     raise TypeError(
        #         "`val_graph` must be a symbolic expression.")
        if key_updates is None:
            key_updates = OrderedDict()
        elif not isinstance(key_updates, OrderedDict):
            raise TypeError("`key_updates` must be an OrderedDict.")
        if val_updates is None:
            val_updates = OrderedDict()
        elif not isinstance(val_updates, OrderedDict):
            raise TypeError("`val_updates` must be an OrderedDict.")
        # Build the list of all inputs
        if isinstance(val_graph, shim.cf.TerminatingTypes):
            val_graphs = [val_graph]
        else:
            val_graphs = val_graph
        all_graphs = (val_graphs + list(val_updates.keys())
                      + list(val_updates.values()))
        inputs = [g for g in shim.graph.symbolic_inputs(all_graphs)
                    if not isinstance(
                        g, (shim.cf.RandomStateType,)+shim.cf.ConstantTypes)]
             # Copying of random state is taken care of separately
        # Check that all non-constant inputs have unique names
        # (we need the names to sub variables back when we reload the graph)
        # We just log the error but don't stop execution, because the code
        # will still run, it just won't benefit from a cache.
        if not all(hasattr(v, 'name') and v.name is not None
                   for v in inputs):
            msg = ("Could not cache graph: All symbolic inputs to a graph "
                   "must have a name before caching it.")
            if self.on_cache_fail == 'warn':
                logger.error(msg)
            elif self.on_cache_fail == 'raise':
                raise CachingError(msg)
            return  # Abort saving to cache

        input_names = [v.name for v in shim.graph.symbolic_inputs(all_graphs)]
        dup_names = [v.name for v, count in Counter(input_names).items()
                     if count > 1]
        if len(dup_names) > 0:
            msg = ("Could not cache graph: multiple symbolic inputs have the "
                   "same name. They need to be unique for caching. Duplicate "
                   "variable names: {}.".format(dup_names))
            if self.on_cache_fail == 'warn':
                logger.error(msg)
            elif self.on_cache_fail == 'raise':
                raise CachingError(msg)
            return  # Abort saving to cache

        with shelve.open(self.cachename) as cache:
            cache = cache[cache['dependencyhashes'][-1]]
                # Latest dependency hash is always the current one
            logger.debug("Caching graph.")
            graphhash = self.get_graph_hash(key_graph)
            updateshash = self.get_graph_hash(key_updates.items())
            try:
                cache[graphhash+updateshash] = (val_graph, val_updates, rng)
            except Exception as e:
                # Don't stop execution just because we weren't able to cache
                # Some graphs aren't picklable, e.g. those containing print ops.
                msg = str(e)
                if self.on_cache_fail == 'warn':
                    logger.error(msg)
                elif self.on_cache_fail == 'raise':
                    raise CachingError(msg).with_traceback(e.__traceback__)

class CompiledGraphCache(GraphCache):
    def get(self, graph, updates=None, other_inputs=None, rng=None):
        if updates is None: updates = OrderedDict()
        with shelve.open(self.cachename) as cache:
            cache = cache[cache['dependencyhashes'][-1]]
                # Latest dependency hash is always the current one
            graphhash = self.get_graph_hash(graph)
            updateshash = self.get_graph_hash(updates.items())
            f = cache.get(graphhash+updateshash, None)
        if f is None:
            logger.debug("CompiledGraphCache miss")
            return None
        else:
            logger.debug("CompiledGraphCache hit")
            f, load_rng = f
            if (rng is None) and (load_rng is not None):
                logger.error("Cannot load graph from cache: cached graph "
                             "requires a random number generator")

        if other_inputs is None:
            other_inputs = []
        elif isinstance(other_inputs, shim.cf.GraphTypes):
            other_inputs = [other_inputs]
        graphs = [graph] if isinstance(graph, shim.cf.GraphTypes) else graph
        # Build the substitution dictionary for the shared variables
        all_graphs = (graphs + list(updates.keys()) + list(updates.values())
                      + list(other_inputs))
        shared_inputs = [si for si in shim.graph.shared_inputs(all_graphs)
                         if not isinstance(si, shim.cf.RandomStateType)]
        shared_inputs_to_sub = [si for si in f.get_shared()
                                if not isinstance(si, shim.cf.RandomStateType)]
        try:
            subdict = self.associate_by_name(shared_inputs_to_sub,
                                             shared_inputs)
        except AttributeError as e:
            logger.error("Cannot load from cache: " + e.args[0])
            return None  # Abort

        # Copy the state of the random number generator
        # FIXME: This leaves the function's RNG disconnected from the model's
        #        RNG. Later changes of the model's seed would have no effect
        #        on the function.
        if load_rng is not None:
            shim.copy_random_state(from_rng=rng, to_rng=load_rng)

        # Construct a wrapper function, which does three things:
        # 1. Set function's shared vars values to those of `shared_inputs`
        # 2. Evaluate the function
        # 3. Transfer the function's shared var values back to `shared_inputs`
        # TODO: Find a way to modify the theano graph directly
        # TODO: Can we use `borrow` for `set_value` as well ?
        def wrapped_f(*args, **kwargs):
            for target, source in subdict.items():
                target.set_value(source.get_value(borrow=True))
            f(*args, **kwargs)
            for target, source in subdict.items():
                source.set_value(target.get_value(borrow=True))

        return wrapped_f

    def set(self, graph, updates, compiled_graph=None, rng=None):
        """
        :param:updates may be omitted: `set(graph, compiled_graph)`
        """
        if isinstance(updates, shim.cf.CompiledType):
            compiled_graph = updates
            updates = OrderedDict()
        elif updates is None:
            updates = OrderedDict()

        # Check that argument types are correct
        if not isinstance(updates, OrderedDict):
            raise TypeError("`updates` must be an OrderedDict.")
        # Build the list of all inputs
        if isinstance(graph, shim.cf.TerminatingTypes):
            graphs = [graph]
        else:
            graphs = graph
        all_graphs = graphs + list(updates.keys()) + list(updates.values())
        shared_inputs = [si for si in shim.graph.shared_inputs(all_graphs)
                            if not isinstance(si, shim.cf.RandomStateType)]
            # Random state is pickled separately and doesn't need varable names
        # Check that all non-constant inputs have unique names
        # (we need the names to sub variables back when we reload the graph)
        # We just log the error but don't stop execution, because the code
        # will still run, it just won't benefit from a cache.
        if not all(hasattr(v, 'name') and v.name is not None
                   for v in shared_inputs):
            logger.error(
                "Could not cache function – all symbolic inputs to a graph "
                "must have a name before caching it.")
            return  # Abort saving to cache

        input_names = [v.name for v in shared_inputs]
        dup_names   = [name for name, count in Counter(input_names).items()
                       if count > 1]
        if len(dup_names) > 0:
            logger.error(
                "Could not cache function – multiple symbolic inputs have the "
                " same name. They need to be unique for caching. Duplicate "
                " variable names: {}.".format(dup_names))
            return  # Abort saving to cache

        with shelve.open(self.cachename) as cache:
            cache = cache[cache['dependencyhashes'][-1]]
                # Latest dependency hash is always the current one
            logger.debug("Caching compiled graph.")
            graphhash = self.get_graph_hash(graph)
            updateshash = self.get_graph_hash(updates.items())
            assert compiled_graph is not None
            try:
                cache[graphhash+updateshash] = (compiled_graph, rng)
            except Exception as e:
                # Don't stop execution just because we weren't able to cache
                # Some graphs aren't picklable, e.g. those containing print ops.
                logger.error(str(e))
