import os
import sys
import random
import string
import warnings
if sys.version_info < (3, 0):
    from cStringIO import StringIO as InMemory
else:
    from io import BytesIO as InMemory
from matplotlib.animation import writers, FileMovieWriter
from base64 import b64encode


ICON_DIR = os.path.join(os.path.dirname(__file__), 'icons')


class _Icons(object):
    """This class is a container for base64 representations of the icons"""
    icons = ['first', 'prev', 'reverse', 'pause', 'play', 'next', 'last']

    def __init__(self, icon_dir=ICON_DIR, extension='png'):
        self.icon_dir = icon_dir
        self.extension = extension
        for icon in self.icons:
            setattr(self, icon,
                    self._load_base64('{0}.{1}'.format(icon, extension)))

    def _load_base64(self, filename):
        data = open(os.path.join(self.icon_dir, filename), 'rb').read()
        return 'data:image/{0};base64,{1}'.format(self.extension,
                                                  b64encode(data).decode('ascii'))


JS_INCLUDE = """
<script language="javascript">
  /* Define the Animation class */
  function Animation(frames, img_id, slider_id, interval, loop_select_id){
    this.img_id = img_id;
    this.slider_id = slider_id;
    this.loop_select_id = loop_select_id;
    this.interval = interval;
    this.current_frame = 0;
    this.direction = 0;
    this.timer = null;
    this.frames = new Array(frames.length);

    for (var i=0; i<frames.length; i++)
    {
     this.frames[i] = new Image();
     this.frames[i].src = frames[i];
    }
    document.getElementById(this.slider_id).max = this.frames.length - 1;
    this.set_frame(this.current_frame);
  }

  Animation.prototype.get_loop_state = function(){
    var button_group = document[this.loop_select_id].state;
    for (var i = 0; i < button_group.length; i++) {
        var button = button_group[i];
        if (button.checked) {
            return button.value;
        }
    }
    return undefined;
  }

  Animation.prototype.set_frame = function(frame){
    this.current_frame = frame;
    document.getElementById(this.img_id).src = this.frames[this.current_frame].src;
    document.getElementById(this.slider_id).value = this.current_frame;
  }

  Animation.prototype.next_frame = function()
  {
    this.set_frame(Math.min(this.frames.length - 1, this.current_frame + 1));
  }

  Animation.prototype.previous_frame = function()
  {
    this.set_frame(Math.max(0, this.current_frame - 1));
  }

  Animation.prototype.first_frame = function()
  {
    this.set_frame(0);
  }

  Animation.prototype.last_frame = function()
  {
    this.set_frame(this.frames.length - 1);
  }

  Animation.prototype.slower = function()
  {
    this.interval /= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
  }

  Animation.prototype.faster = function()
  {
    this.interval *= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
  }

  Animation.prototype.anim_step_forward = function()
  {
    this.current_frame += 1;
    if(this.current_frame < this.frames.length){
      this.set_frame(this.current_frame);
    }else{
      var loop_state = this.get_loop_state();
      if(loop_state == "loop"){
        this.first_frame();
      }else if(loop_state == "reflect"){
        this.last_frame();
        this.reverse_animation();
      }else{
        this.pause_animation();
        this.last_frame();
      }
    }
  }

  Animation.prototype.anim_step_reverse = function()
  {
    this.current_frame -= 1;
    if(this.current_frame >= 0){
      this.set_frame(this.current_frame);
    }else{
      var loop_state = this.get_loop_state();
      if(loop_state == "loop"){
        this.last_frame();
      }else if(loop_state == "reflect"){
        this.first_frame();
        this.play_animation();
      }else{
        this.pause_animation();
        this.first_frame();
      }
    }
  }

  Animation.prototype.pause_animation = function()
  {
    this.direction = 0;
    if (this.timer){
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  Animation.prototype.play_animation = function()
  {
    this.pause_animation();
    this.direction = 1;
    var t = this;
    if (!this.timer) this.timer = setInterval(function(){t.anim_step_forward();}, this.interval);
  }

  Animation.prototype.reverse_animation = function()
  {
    this.pause_animation();
    this.direction = -1;
    var t = this;
    if (!this.timer) this.timer = setInterval(function(){t.anim_step_reverse();}, this.interval);
  }
</script>
"""


DISPLAY_TEMPLATE = """
<div class="animation" align="center">
    <img id="_anim_img{id}">
    <br>
    <input id="_anim_slider{id}" type="range" style="width:350px" name="points" min="0" max="1" step="1" value="0" onchange="anim{id}.set_frame(parseInt(this.value));"></input>
    <br>
    <button onclick="anim{id}.slower()">&#8211;</button>
    <button onclick="anim{id}.first_frame()"><img class="anim_icon" src="{icons.first}"></button>
    <button onclick="anim{id}.previous_frame()"><img class="anim_icon" src="{icons.prev}"></button>
    <button onclick="anim{id}.reverse_animation()"><img class="anim_icon" src="{icons.reverse}"></button>
    <button onclick="anim{id}.pause_animation()"><img class="anim_icon" src="{icons.pause}"></button>
    <button onclick="anim{id}.play_animation()"><img class="anim_icon" src="{icons.play}"></button>
    <button onclick="anim{id}.next_frame()"><img class="anim_icon" src="{icons.next}"></button>
    <button onclick="anim{id}.last_frame()"><img class="anim_icon" src="{icons.last}"></button>
    <button onclick="anim{id}.faster()">+</button>
  <form action="#n" name="_anim_loop_select{id}" class="anim_control">
    <input type="radio" name="state" value="once" {once_checked}> Once </input>
    <input type="radio" name="state" value="loop" {loop_checked}> Loop </input>
    <input type="radio" name="state" value="reflect" {reflect_checked}> Reflect </input>
  </form>
</div>


<script language="javascript">
  /* Instantiate the Animation class. */
  /* The IDs given should match those used in the template above. */
  (function() {{
    var img_id = "_anim_img{id}";
    var slider_id = "_anim_slider{id}";
    var loop_select_id = "_anim_loop_select{id}";
    var frames = new Array({Nframes});
    {fill_frames}

    /* set a timeout to make sure all the above elements are created before
       the object is initialized. */
    setTimeout(function() {{
        anim{id} = new Animation(frames, img_id, slider_id, {interval}, loop_select_id);
    }}, 0);
  }})()
</script>
"""

INCLUDED_FRAMES = """
  for (var i=0; i<{Nframes}; i++){{
    frames[i] = "{frame_dir}/frame" + ("0000000" + i).slice(-7) + ".{frame_format}";
  }}
"""


def _included_frames(frame_list, frame_format):
    """frame_list should be a list of filenames"""
    return INCLUDED_FRAMES.format(Nframes=len(frame_list),
                                  frame_dir=os.path.dirname(frame_list[0]),
                                  frame_format=frame_format)


def _embedded_frames(frame_list, frame_format):
    """frame_list should be a list of base64-encoded png files"""
    template = '  frames[{0}] = "data:image/{1};base64,{2}"\n'
    embedded = "\n"
    for i, frame_data in enumerate(frame_list):
        embedded += template.format(i, frame_format,
                                    frame_data.replace('\n', '\\\n'))
    return embedded


@writers.register('html')
class HTMLWriter(FileMovieWriter):
    # we start the animation id count at a random number: this way, if two
    # animations are meant to be included on one HTML page, there is a
    # very small chance of conflict.
    rng = random.Random()
    exec_key = 'animation.ffmpeg_path'
    args_key = 'animation.ffmpeg_args'
    supported_formats = ['png', 'jpeg', 'tiff', 'svg']

    @classmethod
    def new_id(cls):
        #return '%16x' % cls.rng.getrandbits(64)
        return ''.join(cls.rng.choice(string.ascii_uppercase)
                       for x in range(16))

    def __init__(self, fps=30, codec=None, bitrate=None, extra_args=None,
                 metadata=None, embed_frames=False, default_mode='loop'):
        self.embed_frames = embed_frames
        self.default_mode = default_mode.lower()

        if self.default_mode not in ['loop', 'once', 'reflect']:
            self.default_mode = 'loop'
            warnings.warn("unrecognized default_mode: using 'loop'")

        self._saved_frames = list()
        super(HTMLWriter, self).__init__(fps, codec, bitrate,
                                         extra_args, metadata)

    def setup(self, fig, outfile, dpi, frame_dir=None):
        if os.path.splitext(outfile)[-1] not in ['.html', '.htm']:
            raise ValueError("outfile must be *.htm or *.html")

        if not self.embed_frames:
            if frame_dir is None:
                frame_dir = outfile.rstrip('.html') + '_frames'
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)
            frame_prefix = os.path.join(frame_dir, 'frame')
        else:
            frame_prefix = None

        super(HTMLWriter, self).setup(fig, outfile, dpi,
                                      frame_prefix, clear_temp=False)

    def grab_frame(self, **savefig_kwargs):
        if self.embed_frames:
            suffix = '.' + self.frame_format
            f = InMemory()
            self.fig.savefig(f, format=self.frame_format,
                             dpi=self.dpi, **savefig_kwargs)
            f.seek(0)
            self._saved_frames.append(b64encode(f.read()).decode('ascii'))
        else:
            return super(HTMLWriter, self).grab_frame(**savefig_kwargs)

    def _run(self):
        # make a ducktyped subprocess standin
        # this is called by the MovieWriter base class, but not used here.
        class ProcessStandin(object):
            returncode = 0
            def communicate(self):
                return ('', '')
        self._proc = ProcessStandin()

        # save the frames to an html file
        if self.embed_frames:
            fill_frames = _embedded_frames(self._saved_frames,
                                           self.frame_format)
        else:
            # temp names is filled by FileMovieWriter
            fill_frames = _included_frames(self._temp_names,
                                           self.frame_format)

        mode_dict = dict(once_checked='',
                         loop_checked='',
                         reflect_checked='')
        mode_dict[self.default_mode + '_checked'] = 'checked'

        interval = int(1000. / self.fps)

        with open(self.outfile, 'w') as of:
            of.write(JS_INCLUDE)
            of.write(DISPLAY_TEMPLATE.format(id=self.new_id(),
                                             Nframes=len(self._temp_names),
                                             fill_frames=fill_frames,
                                             interval=interval,
                                             icons=_Icons(),
                                             **mode_dict))
