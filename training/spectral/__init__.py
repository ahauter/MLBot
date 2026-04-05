"""Spectral wavepacket encoding library."""
from training.spectral.wavepacket import (
    WavepacketObject2D,
    compute_feature_maps,
    ConvFeatureExtractor,
    FM_NX, FM_NY, FM_CHANNELS, FM_LABELS,
    COEFF_CLIP, WORLD_BOUNDS,
    COURT_LEFT, COURT_RIGHT, COURT_TOP, COURT_BOTTOM,
    PADDLE_X_OFFSET, PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_SPEED,
    BALL_RADIUS, BALL_SPEED, SPIN_FACTOR,
)

__all__ = [
    'WavepacketObject2D', 'compute_feature_maps', 'ConvFeatureExtractor',
    'FM_NX', 'FM_NY', 'FM_CHANNELS', 'FM_LABELS',
    'COEFF_CLIP', 'WORLD_BOUNDS',
    'COURT_LEFT', 'COURT_RIGHT', 'COURT_TOP', 'COURT_BOTTOM',
    'PADDLE_X_OFFSET', 'PADDLE_WIDTH', 'PADDLE_HEIGHT', 'PADDLE_SPEED',
    'BALL_RADIUS', 'BALL_SPEED', 'SPIN_FACTOR',
]
