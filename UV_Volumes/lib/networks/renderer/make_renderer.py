import imp
import lib.networks.renderer.uv_volumes as uv_volumes

def make_renderer(cfg, network):
    # module = cfg.renderer_module
    # print("render module: " ,module)
    # path = cfg.renderer_path
    # print("render path: " ,path)
    # renderer = imp.load_source(module, path).Renderer(network)
    renderer = uv_volumes.Renderer(network)
    return renderer
