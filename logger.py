"""
https://github.com/cbschaff/pytorch-dl/blob/master/dl/util/logger.py
Change baselines logger to append to log files.
"""
from baselines.logger import *
from baselines.logger import configure as baselines_configure
from tensorboardX import SummaryWriter
import os


def append_human_init(self, filename_or_file):
    if isinstance(filename_or_file, str):
        self.file = open(filename_or_file, 'at')
        self.own_file = True
    else:
        assert hasattr(filename_or_file, 'read'), 'expected file or str, got %s'%filename_or_file
        self.file = filename_or_file
        self.own_file = False

def append_json_init(self, filename):
    self.file = open(filename, 'at')

def append_csv_init(self, filename):
    self.file = open(filename, 'a+t')
    self.keys = []
    self.sep = ','

HumanOutputFormat.__init__ = append_human_init
JSONOutputFormat.__init__ = append_json_init
CSVOutputFormat.__init__ = append_csv_init

# create global tensorboardX summary writer.
WRITER = None
def configure(log_dir, format_strs=None, tbX=False, **kwargs):
    global WRITER
    if tbX:
        tb_dir = os.path.join(log_dir, 'tensorboard')
        WRITER = SummaryWriter(tb_dir, **kwargs)
    else:
        WRITER = None
    baselines_configure(log_dir, format_strs)

def get_summary_writer():
    return WRITER

def add_scalar(tag, scalar_value, global_step=None, walltime=None):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_scalar(tag, scalar_value, global_step, walltime)
    # change interface so both add_scalar and add_scalars adds to the scalar dict.
    WRITER._SummaryWriter__append_to_scalar_dict(tag, scalar_value, global_step, walltime)

def add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

def add_histogram(tag, values, global_step=None, bins='tensorflow'):#, walltime=None, max_bins=None):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_histogram(tag, values, global_step, bins)#, walltime, max_bins)

def add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_image(tag, img_tensor, global_step, walltime, dataformats)

def add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_images(tag, img_tensor, global_step, walltime, dataformats)

def add_image_with_boxes(tag, img_tensor, box_tensor, global_step=None,
                             walltime=None, dataformats='CHW', **kwargs):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_image_with_boxes(tag, img_tensor, box_tensor, global_step,
                                 walltime, dataformats, **kwargs)

def add_figure(tag, figure, global_step=None, close=True, walltime=None):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_figure(tag, figure, global_step, close, walltime)

def add_video(tag, vid_tensor, global_step=None, fps=4, walltime=None):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_video(tag, vid_tensor, global_step, fps, walltime)

def add_audio(tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_audio(tag, snd_tensor, global_step, sample_rate, walltime)

def add_text(tag, text_string, global_step=None, walltime=None):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_text(tag, text_string, global_step, walltime)

def add_graph(model, input_to_model=None, verbose=False, **kwargs):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_graph(model, input_to_model, verbose, **kwargs)


def export_scalars(fname, overwrite=False):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    os.makedirs(os.path.join(WRITER.log_dir, 'scalar_data'), exist_ok=True)
    if fname[-4:] != 'json':
        fname += '.json'
    fname = os.path.join(WRITER.log_dir, 'scalar_data', fname)
    if not os.path.exists(fname) or overwrite:
        WRITER.export_scalars_to_json(fname)
