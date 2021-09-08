"""
MIT License

Copyright (c) 2019 Denis Yarats

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import imageio
import os

import numpy as np


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, camera_id=0, fps=10):
        self.save_dir = make_dir(root_dir, 'video') if root_dir else None
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        # TODO(sff1019): Add dm-control to keep the experiments consistant
        if self.enabled:
            frame = env.render(mode='rgb_array')
            self.frames.append(frame)

    def save(self, file_name, logger=None, step=None):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path,
                            self.frames,
                            fps=self.fps,
                            macro_block_size=None)
        if self.enabled and logger is not None and step is not None:
            # Reshape frames to match pytorch add_video
            h, w, c = self.frames[0].shape
            video_array = np.zeros(shape=(len(self.frames), 3, h, w))
            for idx, frame in enumerate(self.frames):
                video_array[idx, :, :, :] = np.transpose(
                    frame, (2, 0, 1)).astype('uint8')

            logger.log_video(f'eval/{step}',
                             video_array,
                             step,
                             log_frequency=step)
