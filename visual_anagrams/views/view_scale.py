from .view_base import BaseView

class ScaleView(BaseView):
    component_name = "scale"
    def __init__(self, scale=0.5):
        self.scale = scale

    def view(self, im):
        return im

    def inverse_view(self, noise):
        noise[:3] = self.scale * noise[:3]
        return noise
