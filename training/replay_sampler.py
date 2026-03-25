"""
Replay Sampler
==============
Download Rocket League replays from ballchasing.com and convert them into
the token-matrix format used by the shared encoder.

Output layout
-------------
  output_dir/manifest.json          — index of all downloaded replays
  output_dir/raw/<id>.replay        — raw replay files (deleted by default)
  output_dir/parsed/<id>.npz        — parsed token matrices

Each parsed .npz contains:
  tokens : (T, 2, N_TOKENS, TOKEN_FEATURES) float32
             T frames × 2 player perspectives (0=blue, 1=orange)

Labels are kept separate (labels.npz, added later via manual annotation
or KNN auto-labeling) so this file stays pure data with no ML assumptions.

Replay parsing
--------------
Uses rlgym-tools, which wraps carball to produce GameState objects with the
same interface as rlgym-sim. The token construction here mirrors
TokenObsBuilder.build_obs() in training/rlgym_env.py exactly.
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import requests

try:
    from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay
    from rlgym_tools.rocket_league.replays.convert import replay_to_rlgym
except ImportError as _err:
    raise ImportError(
        'rlgym-tools is required for replay parsing.\n'
        'Install it with: pip install rlgym-tools'
    ) from _err

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from encoder import (
    FIELD_X, FIELD_Y, CEILING_Z,
    MAX_VEL, MAX_ANG_VEL, MAX_BOOST,
    N_TOKENS, TOKEN_FEATURES,
)

# ── boost pad layout (must match training/rlgym_env.py) ───────────────────────
_BIG_PAD_INDICES = [3, 4, 15, 18, 29, 30]
_BIG_PAD_POSITIONS = np.array([
    [-3584.0,     0.0, 73.0],
    [ 3584.0,     0.0, 73.0],
    [-3072.0,  4096.0, 73.0],
    [ 3072.0,  4096.0, 73.0],
    [-3072.0, -4096.0, 73.0],
    [ 3072.0, -4096.0, 73.0],
], dtype=np.float32)

BALLCHASING_API = 'https://ballchasing.com/api'

# Ordered from lowest to highest so CLI choices validation works cleanly
VALID_RANKS = [
    'unranked',
    'bronze-1', 'bronze-2', 'bronze-3',
    'silver-1', 'silver-2', 'silver-3',
    'gold-1', 'gold-2', 'gold-3',
    'platinum-1', 'platinum-2', 'platinum-3',
    'diamond-1', 'diamond-2', 'diamond-3',
    'champion-1', 'champion-2', 'champion-3',
    'grand-champion-1', 'grand-champion-2', 'grand-champion-3',
    'supersonic-legend',
]


# ── token construction ────────────────────────────────────────────────────────

def gamestate_to_tokens(state, player_idx: int) -> np.ndarray:
    """
    Convert an rlgym-sim-compatible GameState to (N_TOKENS, TOKEN_FEATURES) float32.

    Mirrors TokenObsBuilder.build_obs() in training/rlgym_env.py exactly,
    including the opponent-boost-hidden convention and the big-pad index list.

    Parameters
    ----------
    state       : rlgym-sim GameState (from rlgym-tools replay parser)
    player_idx  : 0 = encode from blue's perspective, 1 = orange's perspective
    """
    own_player: Optional[object] = None
    opp_player: Optional[object] = None
    for car in state.cars.values():
        if car.team_num == player_idx and own_player is None:
            own_player = car
        elif car.team_num != player_idx and opp_player is None:
            opp_player = car

    ball = state.ball

    # ── token 0: ball ─────────────────────────────────────────────────────────
    ball_tok = np.array([
        ball.position[0]         / FIELD_X,
        ball.position[1]         / FIELD_Y,
        ball.position[2]         / CEILING_Z,
        ball.linear_velocity[0]  / MAX_VEL,
        ball.linear_velocity[1]  / MAX_VEL,
        ball.linear_velocity[2]  / MAX_VEL,
        ball.angular_velocity[0] / MAX_ANG_VEL,
        ball.angular_velocity[1] / MAX_ANG_VEL,
        ball.angular_velocity[2] / MAX_ANG_VEL,
        0.0,
    ], dtype=np.float32)

    # ── token 1: own car ──────────────────────────────────────────────────────
    if own_player is not None:
        own = own_player.physics
        own_boost = own_player.boost_amount   # rlgym Car: 0..100 scale
        own_tok = np.array([
            own.position[0]        / FIELD_X,
            own.position[1]        / FIELD_Y,
            own.position[2]        / CEILING_Z,
            own.linear_velocity[0] / MAX_VEL,
            own.linear_velocity[1] / MAX_VEL,
            own.linear_velocity[2] / MAX_VEL,
            own.yaw               / math.pi,
            own.pitch             / math.pi,
            own.roll              / math.pi,
            own_boost             / MAX_BOOST,
        ], dtype=np.float32)
    else:
        own_tok = np.zeros(TOKEN_FEATURES, dtype=np.float32)

    # ── token 2: opponent — boost intentionally hidden ────────────────────────
    if opp_player is not None:
        opp = opp_player.physics
        opp_tok = np.array([
            opp.position[0]        / FIELD_X,
            opp.position[1]        / FIELD_Y,
            opp.position[2]        / CEILING_Z,
            opp.linear_velocity[0] / MAX_VEL,
            opp.linear_velocity[1] / MAX_VEL,
            opp.linear_velocity[2] / MAX_VEL,
            opp.yaw               / math.pi,
            opp.pitch             / math.pi,
            opp.roll              / math.pi,
            0.0,   # opponent boost not observable in real matches
        ], dtype=np.float32)
    else:
        opp_tok = np.zeros(TOKEN_FEATURES, dtype=np.float32)

    # ── tokens 3-8: big boost pads ────────────────────────────────────────────
    pad_toks = []
    for i, idx in enumerate(_BIG_PAD_INDICES):
        active = float(state.boost_pad_timers[idx] == 0) if idx < len(state.boost_pad_timers) else 0.0
        pos = _BIG_PAD_POSITIONS[i]
        pad_toks.append(np.array([
            pos[0] / FIELD_X,
            pos[1] / FIELD_Y,
            pos[2] / CEILING_Z,
            active,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32))

    # ── token 9: game state ───────────────────────────────────────────────────
    # Score and time remaining are not reliably available through the GameState
    # interface when parsing replays, so we leave them zeroed — consistent with
    # how TokenObsBuilder handles the sim environment.
    gs_tok = np.zeros(TOKEN_FEATURES, dtype=np.float32)

    tokens = np.stack([ball_tok, own_tok, opp_tok] + pad_toks + [gs_tok], axis=0)
    return tokens   # (N_TOKENS, TOKEN_FEATURES)


# ── replay parsing ────────────────────────────────────────────────────────────

def parse_replay(replay_path: Path) -> np.ndarray:
    """
    Parse a .replay file into a token-matrix array.

    Returns
    -------
    np.ndarray of shape (T, 2, N_TOKENS, TOKEN_FEATURES) float32
      T   : number of frames
      2   : player perspectives — index 0 = blue, index 1 = orange
    """
    replay = ParsedReplay.load(str(replay_path))
    replay_frames = list(replay_to_rlgym(replay))

    if not replay_frames:
        raise ValueError(f'No frames parsed from {replay_path}')

    frames = []
    for replay_frame in replay_frames:
        state = replay_frame.state
        blue_tok   = gamestate_to_tokens(state, player_idx=0)
        orange_tok = gamestate_to_tokens(state, player_idx=1)
        frames.append(np.stack([blue_tok, orange_tok], axis=0))   # (2, N_TOKENS, F)

    return np.stack(frames, axis=0).astype(np.float32)   # (T, 2, N_TOKENS, F)


# ── ballchasing.com API client ────────────────────────────────────────────────

class BallchasingClient:
    """
    Thin wrapper around the ballchasing.com REST API.

    API key is obtained from https://ballchasing.com (free account required).
    Pass it via the Authorization header as a plain token string.
    """

    def __init__(self, api_key: str) -> None:
        self._session = requests.Session()
        self._session.headers['Authorization'] = api_key

    def search_replays(
        self,
        count: int,
        min_rank: str = 'bronze-1',
        max_rank: str = 'bronze-3',
        playlist: str = 'ranked-standard',
    ) -> list[dict]:
        """
        Return metadata dicts for up to `count` replays matching the rank/playlist filter.
        Paginates automatically (ballchasing allows max 200 results per request).
        """
        results: list[dict] = []
        next_url: Optional[str] = None
        page_size = min(count, 200)

        params = {
            'min-rank':  min_rank,
            'max-rank':  max_rank,
            'playlist':  playlist,
            'count':     page_size,
            'sort-by':   'replay-date',
            'sort-dir':  'desc',
        }

        while len(results) < count:
            if next_url:
                resp = self._session.get(next_url)
            else:
                resp = self._session.get(f'{BALLCHASING_API}/replays', params=params)
            resp.raise_for_status()
            data = resp.json()
            results.extend(data.get('list', []))
            next_url = data.get('next')
            if not next_url:
                break

        return results[:count]

    def download_replay(self, replay_id: str, dest_path: Path) -> Path:
        """Stream-download a .replay binary to dest_path."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with self._session.get(
            f'{BALLCHASING_API}/replays/{replay_id}/file', stream=True
        ) as resp:
            resp.raise_for_status()
            with open(dest_path, 'wb') as f:
                shutil.copyfileobj(resp.raw, f)
        return dest_path


# ── manifest helpers ──────────────────────────────────────────────────────────

def _load_manifest(manifest_path: Path) -> dict[str, dict]:
    """Load manifest.json and return a {replay_id: entry} dict."""
    if manifest_path.exists():
        with open(manifest_path) as f:
            entries = json.load(f)
        return {e['replay_id']: e for e in entries}
    return {}


def _save_manifest(manifest_path: Path, manifest: dict[str, dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(list(manifest.values()), f, indent=2)


# ── main pipeline ─────────────────────────────────────────────────────────────

def collect(
    api_key: str,
    count: int,
    output_dir: Path,
    min_rank: str = 'bronze-1',
    max_rank: str = 'bronze-3',
    playlist: str = 'ranked-standard',
    keep_raw: bool = False,
    resume: bool = True,
) -> None:
    """
    Download, parse, and save `count` replays.

    Parameters
    ----------
    api_key     : ballchasing.com API key
    count       : number of replays to collect
    output_dir  : root directory for all output
    min_rank    : lower bound of rank filter (inclusive)
    max_rank    : upper bound of rank filter (inclusive)
    playlist    : ballchasing playlist identifier
    keep_raw    : if True, keep .replay files after parsing
    resume      : if True, skip replay IDs already present in manifest.json
    """
    output_dir    = Path(output_dir)
    raw_dir       = output_dir / 'raw'
    parsed_dir    = output_dir / 'parsed'
    manifest_path = output_dir / 'manifest.json'

    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(manifest_path) if resume else {}
    client   = BallchasingClient(api_key)

    print(f'Searching ballchasing.com for {count} replays '
          f'(rank: {min_rank}–{max_rank}, playlist: {playlist}) ...')
    replay_list = client.search_replays(count, min_rank, max_rank, playlist)
    print(f'Found {len(replay_list)} replays from API.')

    saved   = 0
    skipped = 0
    failed  = 0

    for meta in replay_list:
        rid = meta['id']

        if resume and rid in manifest:
            skipped += 1
            continue

        raw_path    = raw_dir    / f'{rid}.replay'
        parsed_path = parsed_dir / f'{rid}.npz'

        print(f'  [{saved + skipped + 1}/{len(replay_list)}] {rid} ...', end=' ', flush=True)

        try:
            client.download_replay(rid, raw_path)
            tokens = parse_replay(raw_path)              # (T, 2, N_TOKENS, TOKEN_FEATURES)
            np.savez_compressed(parsed_path, tokens=tokens)
        except Exception as exc:
            print(f'FAILED ({exc})')
            if raw_path.exists():
                raw_path.unlink()
            failed += 1
            continue

        if not keep_raw:
            raw_path.unlink()

        manifest[rid] = {
            'replay_id':     rid,
            'rank':          meta.get('min_rank', {}).get('name', min_rank),
            'playlist':      meta.get('playlist_id', playlist),
            'frame_count':   int(tokens.shape[0]),
            'parsed_path':   str(parsed_path.relative_to(output_dir)),
            'downloaded_at': datetime.now(timezone.utc).isoformat(),
        }
        _save_manifest(manifest_path, manifest)

        print(f'OK  ({tokens.shape[0]} frames)')
        saved += 1

    print(f'\nDone. Saved {saved} | Skipped {skipped} | Failed {failed}')
    print(f'Total in manifest: {len(manifest)}')
