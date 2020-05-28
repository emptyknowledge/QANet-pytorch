# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
#
# cython: language_level=3
#

def visual_tensorboard(log_dir, tag, data, epoch, step):
  from tensorboardX import SummaryWriter
  for name, value in data.items():
    if value is None or len(value) < 1:
      # config.logger.warning(f"name: {name} Gradient is null")
      continue
    elif len(value) > 0:
      writer = SummaryWriter(f"{log_dir}/{name}")
      writer.add_histogram(f"{epoch}_{tag}", value, step)
      writer.close()

if __name__ == "__main__":
  pass

