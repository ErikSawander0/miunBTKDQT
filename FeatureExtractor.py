"""
file: ./FeatureExtractor.py
"""
class FeatureExtractor:
    def __init__(self, model, layer_indices: list[int]):
        self.features = {}
        self.hooks = []
        
        for idx in layer_indices:
            layer = model.backbone.encoder.layer[idx]
            hook = layer.register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, idx):
        def hook(module, input, output): 
            self.features[idx] = output[0] if isinstance(output, tuple) else output
        return hook
    
    def clear(self):
        self.features = {}
    
    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
