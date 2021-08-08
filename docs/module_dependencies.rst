Dependencies between *mackelab_toolbox* modules
-----------------------------------------------

The *Mackelab Toolbox* is meant to provide standard solutions to recurring coding problems in our field. This philosophy extends to the toolbox itself, so toolbox functions make liberal use of other toolbox functions. For users, this provides all the benefits of shared code, in terms of consistency and reliability. For developers, these benefits still apply, but there is a small price to pay: one must take care not to introduce import cycles (A imports B and B imports A). The solution is to have a clear ordering between modules, and never import a “downstream” module into an “upstream” one.  The diagram below shows this dependency structure at the time of writing. (Note that while intimidating, this diagram is meant to be used piecemeal: if one is working on the `units` module, the one can see from this diagram that `typing` should not be imported, since it is downstream.)

.. mermaid::
   :caption: Dependencies between *mackelab_toolbox* modules. An arrow from modules A to B indicates that A imports B. Modules are grouped into *contexts*, indicated by yellow boxes. Dependencies for individual contexts can be install by providing their name as an extra argument to `pip install`. (Some contexts depend on others, which are then also installed.)

   graph BT
     subgraph utils_context["utils"]
       meta
       utils --> meta
     end
     
     subgraph typing_context["typing & serialization"]
       typing
       typing_module
       typing_pure_function
       serialize
       pydantic
     end
     typing --> typing_module
     typing_module --> utils
     typing_module --> units
     typing_pure_function --> typing_module
     typing_pure_function --> serialize
     serialize --> utils
     
     subgraph stats
       transform
       transform_postponed
     end
     transform --> typing
     transform --> pydantic
     transform --> utils
     transform_postponed --> transform
     
     subgraph pymc_context["pymc3"]
       pymc3
       pymc_typing
     end
     pymc3 --> transform
     pymc3 --> iotools
     pymc3 --> utils
     pymc3 --> parameters
     
     subgraph theano_context["theano"]
       theano
       cgshim_postponed
       cgshim
     end
     pymc_typing --> theano
     pymc_typing --> typing
     
     subgraph iotools_context["iotools"]
       iotools
     end
     subgraph parameters_context["parameters"]
       parameters
     end
     
     cgshim_postponed --> typing
     cgshim_postponed --> cgshim
     cgshim --> typing
     cgshim --> utils
     cgshim --> transform
     
     rcParams
     optimizers
     parameters
     subgraph tqdm_context["tqdm"]
       tqdm
     end
     units --> utils
     
     subgraph plotting_context["plotting"]
       plot --> colors
       plot --> utils
       plot --> rcParams
     end
     
     subgraph torch_context["torch"]
       torch
     end
     torch --> iotools
     torch --> utils
