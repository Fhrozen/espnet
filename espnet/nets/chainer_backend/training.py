"""Schedulers for Training."""
from __future__ import division

# chainer related
from chainer.training import extension

import numpy as np

def _schedule_invsqrt():
    return

class Scheduler(extension.Extension):
    """Trainer extension to shift an optimizer attribute.

    Args:
        attr (str): Name of the attribute to shift.
        rate (float): Rate of the exponential shift. This value is multiplied
            to the attribute at each call.
        init (float): Initial value of the attribute. If it is ``None``, the
            extension extracts the attribute at the first call and uses it as
            the initial value.
        target (float): Target value of the attribute. If the attribute reaches
            this value, the shift stops.
        optimizer (~chainer.Optimizer): Target optimizer to adjust the
            attribute. If it is ``None``, the main optimizer of the updater is
            used.

    """

    def __init__(self, attr, d, warmup_steps=4000,
                 init=None, target=None, optimizer=None,
                 scale=1., schedule='inv_sqrt'):
        """Initialize Vaswani rule extension."""
        self._attr = attr
        self._d_inv05 = d ** (-0.5) * scale
        self._warmup_steps_inv15 = warmup_steps ** (-1.5)
        self._init = init
        self._target = target
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None
        if schedule == 'inv_sqrt':
            self.schedule = self._schedule_invsqrt
        else:
            raise ValueError(f'The selected scheduler is not correct: {schedule}')

    def initialize(self, trainer):
        """Initialize Optimizer values."""
        optimizer = self._get_optimizer(trainer)
        # ensure that _init is set
        if self._init is None:
            self._init = self._d_inv05 * (1. * self._warmup_steps_inv15)
        if self._last_value is not None:  # resuming from a snapshot
            self._update_value(optimizer, self._last_value)
        else:
            self._update_value(optimizer, self._init)

    def __call__(self, trainer):
        """Forward extension."""
        self._t += 1
        optimizer = self._get_optimizer(trainer)
        value = self.schedule()
        self._update_value(optimizer, value)
    
    def _schedule_invsqrt(self):
        return self._d_inv05 * \
            min(self._t ** (-0.5), self._t * self._warmup_steps_inv15)

    def serialize(self, serializer):
        """Serialize extension."""
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)

    def _get_optimizer(self, trainer):
        """Obtain optimizer from trainer."""
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        """Update requested variable values."""
        setattr(optimizer, self._attr, value)
        self._last_value = value

