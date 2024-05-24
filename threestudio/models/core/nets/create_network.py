import imp

from omegaconf import OmegaConf
config_file = "/home/yjli/AIGC/threestudio/configs/humannerf.yaml"
cfg = OmegaConf.load(config_file)

def _query_network():
    module = cfg.network_module
    module_path = module.replace(".", "/") + ".py"
    network = imp.load_source(module, module_path).Network
    return network


def create_network():
    network = _query_network()
    network = network()
    return network
