import numpy   # To avoid bugs, it's best to ensure numpy is imported before torch
import torch
import mackelab_toolbox.iotools as io
import mackelab_toolbox.utils as utils

# Register pytorch format
ioformat = io.Format('torchstate',
                     save=lambda f,data: torch.save(data, f),
                     load=torch.load,
                     bytes=True)
io.defined_formats['torchstate'] = ioformat
# io.register_datatype(nn.Module, format=ioformat)
#   -> the ioformat datatype is actually dict

utils._terminating_types |= {numpy.number, numpy.ndarray, torch.Tensor}
utils.terminating_types = tuple(utils._terminating_types)  # NB: isinstance() only works with tuples
