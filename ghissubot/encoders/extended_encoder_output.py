from collections import namedtuple

from seq2seq.encoders.encoder import EncoderOutput

"""Adds a field to EncoderOutput to include context encoder output"""
ExtendedEncoderOutput = namedtuple(
    "ExtendedEncoderOutput",
    EncoderOutput._fields + ("context_outputs",))