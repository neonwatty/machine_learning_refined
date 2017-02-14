from .html_writer import HTMLWriter
from matplotlib.animation import Animation
import matplotlib.pyplot as plt
import tempfile
import random
import os


__all__ = ['anim_to_html', 'display_animation']


class _NameOnlyTemporaryFile(object):
    """A context-managed temporary file which is not opened.

    The file should be accessible by name on any system.

    Parameters
    ----------
    suffix : string
        The suffix of the temporary file (default = '')
    prefix : string
        The prefix of the temporary file (default = '_tmp_')
    hash_length : string
        The length of the random hash.  The size of the hash space will
        be 16 ** hash_length (default=8)
    seed : integer
        the seed for the random number generator.  If not specified, the
        system time will be used as a seed.
    absolute : boolean
        If true, return an absolute path to a temporary file in the current
        working directory.

    Example
    -------

    >>> with _NameOnlyTemporaryFile(seed=0, absolute=False) as f:
    ...     print(f)
    ...
    _tmp_d82c07cd
    >>> os.path.exists('_tmp_d82c07cd')  # file removed after context
    False

    """
    def __init__(self, prefix='_tmp_', suffix='', hash_length=8,
                 seed=None, absolute=True):
        rng = random.Random(seed)
        self.name = '%s%0*x%s' % (prefix, hash_length,
                                  rng.getrandbits(4 * hash_length), suffix)
        if absolute:
            self.name = os.path.abspath(self.name)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        if os.path.exists(self.name):
            os.remove(self.name)


def anim_to_html(anim, fps=None, embed_frames=True, default_mode='loop'):
    """Generate HTML representation of the animation"""
    if fps is None and hasattr(anim, '_interval'):
        # Convert interval in ms to frames per second
        fps = 1000. / anim._interval

    plt.close(anim._fig)
    if hasattr(anim, "_html_representation"):
        return anim._html_representation
    else:
        # tempfile can't be used here: we need a filename, and this
        # fails on windows.  Instead, we use a custom filename generator
        #with tempfile.NamedTemporaryFile(suffix='.html') as f:
        with _NameOnlyTemporaryFile(suffix='.html') as f:
            anim.save(f.name,  writer=HTMLWriter(fps=fps,
                                                 embed_frames=embed_frames,
                                                 default_mode=default_mode))
            html = open(f.name).read()

        anim._html_representation = html
        return html


def display_animation(anim, **kwargs):
    """Display the animation with an IPython HTML object"""
    from IPython.display import HTML
    return HTML(anim_to_html(anim, **kwargs))


# This is the magic that makes animations display automatically in the
# IPython notebook.  The _repr_html_ method is a special method recognized
# by IPython.
Animation._repr_html_ = anim_to_html
