import traitlets


_ERR_MESSAGE = "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"

# patched version which allows for numpy comparison
# https://github.com/jupyter-widgets/traittypes/issues/45
class link_widgets(traitlets.link):

    def _update_target(self, change):
        try:
            super()._update_target(change)
        except ValueError as e:
            if e.args[0] != _ERR_MESSAGE:
                raise e

    def _update_source(self, change):
        try:
            super()._update_source(change)
        except ValueError as e:
            if e.args[0] != _ERR_MESSAGE:
                raise e
