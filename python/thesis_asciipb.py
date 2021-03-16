from google.protobuf.text_format import PrintMessage
from proto.py import algorithm_params_pb2
from proto.py import distribution_pb2
from proto.py import hierarchy_prior_pb2
from proto.py import mixing_prior_pb2

# Run this from root with python -m python.thesis_asciipb

def identity_list(dim):
  """Returns the list of entries of a dim-dimensional identity matrix."""
  ide = dim*dim*[0.0]
  for i in range(0, dim*dim, dim+1):
    ide[i] = 1.0
  return ide

if __name__ == '__main__':
  ASCIIPB = 'resources/asciipb/thesis/'
  ## ALGORITHM PARAMETERS
  settings = algorithm_params_pb2.AlgorithmParams()
  settings.rng_seed = 20201124
  settings.iterations = 1000
  settings.burnin = 100
  settings.init_num_clusters = 10
  settings.neal8_n_aux = 3
  for algo in ['Neal2', 'Neal3', 'Neal8', 'BlockedGibbs']:
    settings.algo_id = algo
    with open(ASCIIPB + algo + '.asciipb', 'w') as f:
      PrintMessage(settings, f)

  ## MIXING PRIORS
  # For "galaxy" and "faithful": DP and PY mixings
  dp_prior = mixing_prior_pb2.DPPrior()
  dp_prior.gamma_prior.totalmass_prior.shape = 4.0
  dp_prior.gamma_prior.totalmass_prior.rate = 2.0
  with open(ASCIIPB + 'DP.asciipb', 'w') as f:
    PrintMessage(dp_prior, f)
  py_prior = mixing_prior_pb2.PYPrior()
  py_prior.fixed_values.strength = 1.0
  py_prior.fixed_values.discount = 0.1
  with open(ASCIIPB + 'PY.asciipb', 'w') as f:
    PrintMessage(py_prior, f)
  # For "dde": LSB and TSB mixings
  dim = 1
  mu00 = dim*[0.0]
  sig00 = [5.0*_ for _ in identity_list(dim)]
  lsb_prior = mixing_prior_pb2.LogSBPrior()
  lsb_prior.normal_prior.mean.size = dim
  lsb_prior.normal_prior.mean.data[:] = mu00
  lsb_prior.normal_prior.var.rows = dim
  lsb_prior.normal_prior.var.cols = dim
  lsb_prior.normal_prior.var.data[:] = sig00
  lsb_prior.step_size = 0.025
  lsb_prior.num_components = 4
  with open(ASCIIPB + 'LogSB.asciipb', 'w') as f:
    PrintMessage(lsb_prior, f)
  truncsb_prior_dp = mixing_prior_pb2.TruncSBPrior()
  truncsb_prior_dp.num_components = 4
  truncsb_prior_dp.dp_prior.totalmass = 0.5
  with open(ASCIIPB + 'TruncSB.asciipb', 'w') as f:
    PrintMessage(truncsb_prior_dp, f)

  ## HIERARCHY PRIORS
  # For "galaxy": NGG hyperprior for NNIG hierarchy
  nnig_prior = hierarchy_prior_pb2.NNIGPrior()
  nnig_prior.ngg_prior.mean_prior.mean = 25.0  # sample mean = 20.82
  nnig_prior.ngg_prior.mean_prior.var = 4.0
  nnig_prior.ngg_prior.var_scaling_prior.shape = 0.4
  nnig_prior.ngg_prior.var_scaling_prior.rate = 0.2
  nnig_prior.ngg_prior.shape = 4.0
  nnig_prior.ngg_prior.scale_prior.shape = 4.0
  nnig_prior.ngg_prior.scale_prior.rate = 2.0
  with open(ASCIIPB + 'hier_galaxy.asciipb', 'w') as f:
    PrintMessage(nnig_prior, f)
  # For "faithful": NGIW hyperprior for NNW hierarchy
  nnw_prior = hierarchy_prior_pb2.NNWPrior()
  dim = 2
  mu00 = dim * [3.0]  # sample mean = [3.48, 3.48]
  ident = identity_list(dim)
  nu0 = 4.0
  nnw_prior.ngiw_prior.mean_prior.mean.size = dim
  nnw_prior.ngiw_prior.mean_prior.mean.data[:] = mu00
  sig00 = [i/nu0 for i in ident]
  nnw_prior.ngiw_prior.mean_prior.var.rows = dim
  nnw_prior.ngiw_prior.mean_prior.var.cols = dim
  nnw_prior.ngiw_prior.mean_prior.var.data[:] = sig00
  nnw_prior.ngiw_prior.mean_prior.var.rowmajor = False
  nnw_prior.ngiw_prior.var_scaling_prior.shape = 0.4
  nnw_prior.ngiw_prior.var_scaling_prior.rate = 0.2
  nnw_prior.ngiw_prior.deg_free = nu0
  nnw_prior.ngiw_prior.scale_prior.deg_free = nu0
  tau00 = [i*nu0 for i in ident]
  nnw_prior.ngiw_prior.scale_prior.scale.rows = dim
  nnw_prior.ngiw_prior.scale_prior.scale.cols = dim
  nnw_prior.ngiw_prior.scale_prior.scale.data[:] = tau00
  nnw_prior.ngiw_prior.scale_prior.scale.rowmajor = False
  with open(ASCIIPB + 'hier_faithful.asciipb', 'w') as f:
    PrintMessage(nnw_prior, f)
  # For "dde": NGG hyperprior for NNIG hierarchy
  nnig_prior = hierarchy_prior_pb2.NNIGPrior()
  nnig_prior.ngg_prior.mean_prior.mean = 25.0  # sample mean = 30.24
  nnig_prior.ngg_prior.mean_prior.var = 4.0
  nnig_prior.ngg_prior.var_scaling_prior.shape = 0.4
  nnig_prior.ngg_prior.var_scaling_prior.rate = 0.2
  nnig_prior.ngg_prior.shape = 4.0
  nnig_prior.ngg_prior.scale_prior.shape = 4.0
  nnig_prior.ngg_prior.scale_prior.rate = 2.0
  with open(ASCIIPB + 'hier_dde.asciipb', 'w') as f:
    PrintMessage(nnig_prior, f)