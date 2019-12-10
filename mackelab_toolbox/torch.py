import torch
import mackelab_toolbox.iotools as io

# Register pytorch format
ioformat = io.Format('torchstate',
                     save=lambda f,data: torch.save(data, f),
                     load=torch.load,
                     bytes=True)
io.defined_formats['torchstate'] = ioformat
# io.register_datatype(nn.Module, format=ioformat)
#   -> the ioformat datatype is actually dict
