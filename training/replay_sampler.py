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
  tokens  : (T, 2, N_TOKENS, TOKEN_FEATURES) float32
  actions : (T, 2, 8)                        float32
  rewards : (T, 2)                           float32  — sparse +1/-1 at goal frames
  dones   : (T, 2)                           bool     — True on last frame of each episode

Replay parsing
--------------
Uses rlgym-tools, which produces ReplayFrame objects with rlgym.rocket_league.api.GameState.
Note: this is a DIFFERENT GameState than rlgym-sim — key differences:
  state.cars            : Dict[AgentID, Car]      (not state.players)
  car.physics           : PhysicsObject           (not car.car_data)
  car.boost_amount      : float 0..100            (not 0..1)
  physics.yaw/.pitch/.roll : properties           (not methods)
  state.boost_pad_timers: np.ndarray              (0 = active, >0 = respawning)
"""

from __future__ import annotations
from encoder import (
    FIELD_X, FIELD_Y, CEILING_Z,
    MAX_VEL, MAX_ANG_VEL, MAX_BOOST,
    N_TOKENS, TOKEN_FEATURES,
)

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

# ── boost pad layout (must match training/rlgym_env.py) ───────────────────────
_BIG_PAD_INDICES = [3, 4, 15, 18, 29, 30]
_BIG_PAD_POSITIONS = np.array([
    [-3584.0,     0.0, 73.0],
    [3584.0,     0.0, 73.0],
    [-3072.0,  4096.0, 73.0],
    [3072.0,  4096.0, 73.0],
    [-3072.0, -4096.0, 73.0],
    [3072.0, -4096.0, 73.0],
], dtype=np.float32)

BALLCHASING_API = 'https://ballchasing.com/api'

# terminal threshold: < 2 ticks at 30fps → last frame of an episode
_GOAL_THRESHOLD = 1.0 / 15.0

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
    Convert an rlgym.rocket_league.api.GameState to (N_TOKENS, TOKEN_FEATURES) float32.

    Uses the rlgym (replay) API — NOT rlgym-sim:
      state.cars               Dict[AgentID, Car]
      car.physics              PhysicsObject
      car.boost_amount         0..100
      physics.yaw/.pitch/.roll properties (no parentheses)
      state.boost_pad_timers   0 = active, >0 = respawning

    Parameters
    ----------
    state       : rlgym.rocket_league.api.GameState
    player_idx  : 0 = blue perspective, 1 = orange perspective
    """
    own_car: Optional[object] = None
    opp_car: Optional[object] = None
    for car in state.cars.values():
        if car.team_num == player_idx and own_car is None:
            own_car = car
        elif car.team_num != player_idx and opp_car is None:
            opp_car = car

    ball = state.ball

    # ── token 0: ball ─────────────────────────────────────────────────────────
    ball_tok = np.array([
        ball.position[0] / FIELD_X,
        ball.position[1] / FIELD_Y,
        ball.position[2] / CEILING_Z,
        ball.linear_velocity[0] / MAX_VEL,
        ball.linear_velocity[1] / MAX_VEL,
        ball.linear_velocity[2] / MAX_VEL,
        ball.angular_velocity[0] / MAX_ANG_VEL,
        ball.angular_velocity[1] / MAX_ANG_VEL,
        ball.angular_velocity[2] / MAX_ANG_VEL,
        0.0,
    ], dtype=np.float32)

    # ── token 1: own car ──────────────────────────────────────────────────────
    if own_car is not None:
        own = own_car.physics
        own_boost = own_car.boost_amount   # 0..100 in rlgym Car
        own_tok = np.array([
            own.position[0] / FIELD_X,
            own.position[1] / FIELD_Y,
            own.position[2] / CEILING_Z,
            own.linear_velocity[0] / MAX_VEL,
            own.linear_velocity[1] / MAX_VEL,
            own.linear_velocity[2] / MAX_VEL,
            own.yaw / math.pi,
            own.pitch / math.pi,
            own.roll / math.pi,
            own_boost / MAX_BOOST,
        ], dtype=np.float32)
    else:
        own_tok = np.zeros(TOKEN_FEATURES, dtype=np.float32)

    # ── token 2: opponent — boost intentionally hidden ────────────────────────
    if opp_car is not None:
        opp = opp_car.physics
        opp_tok = np.array([
            opp.position[0] / FIELD_X,
            opp.position[1] / FIELD_Y,
            opp.position[2] / CEILING_Z,
            opp.linear_velocity[0] / MAX_VEL,
            opp.linear_velocity[1] / MAX_VEL,
            opp.linear_velocity[2] / MAX_VEL,
            opp.yaw / math.pi,
            opp.pitch / math.pi,
            opp.roll / math.pi,
            0.0,   # opponent boost not observable in real matches
        ], dtype=np.float32)
    else:
        opp_tok = np.zeros(TOKEN_FEATURES, dtype=np.float32)

    # ── tokens 3-8: big boost pads ────────────────────────────────────────────
    pad_toks = []
    for i, idx in enumerate(_BIG_PAD_INDICES):
        # boost_pad_timers: 0 = pad is active, >0 = seconds until respawn
        active = float(state.boost_pad_timers[idx] == 0) if idx < len(
            state.boost_pad_timers) else 0.0
        pos = _BIG_PAD_POSITIONS[i]
        pad_toks.append(np.array([
            pos[0] / FIELD_X,
            pos[1] / FIELD_Y,
            pos[2] / CEILING_Z,
            active,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32))

    # ── token 9: game state (score/time not reliably available) ───────────────
    gs_tok = np.zeros(TOKEN_FEATURES, dtype=np.float32)

    tokens = np.stack([ball_tok, own_tok, opp_tok] +
                      pad_toks + [gs_tok], axis=0)
    return tokens   # (N_TOKENS, TOKEN_FEATURES)


# ── replay parsing ────────────────────────────────────────────────────────────

def parse_replay(replay_path: Path):
    """
    Parse a .replay file into token, action, reward, and done arrays.

    Returns
    -------
    tokens  : (T, 2, N_TOKENS, TOKEN_FEATURES) float32
    actions : (T, 2, 8)  float32  — [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
    rewards : (T, 2)     float32  — sparse ±1 at goal frames, 0 elsewhere
    dones   : (T, 2)     bool     — True on the terminal frame of each episode
    """
    replay = ParsedReplay.load(str(replay_path))
    replay_frames = list(replay_to_rlgym(replay))

    if not replay_frames:
        raise ValueError(f'No frames parsed from {replay_path}')

    frames_tokens = []
    frames_actions = []
    frames_rewards = []
    frames_dones = []

    for replay_frame in replay_frames:
        state = replay_frame.state

        # agent_id → team_num mapping
        agent_team = {aid: car.team_num for aid, car in state.cars.items()}

        # collect actions keyed by team (0=blue, 1=orange)
        team_action = {0: np.zeros(8, np.float32), 1: np.zeros(8, np.float32)}
        for aid, act in replay_frame.actions.items():
            team = agent_team.get(aid)
            if team is not None:
                team_action[team] = np.asarray(act, dtype=np.float32)[:8]

        # terminal frame detection + sparse reward
        is_done = replay_frame.episode_seconds_remaining < _GOAL_THRESHOLD
        scorer = replay_frame.next_scoring_team   # 0=blue, 1=orange, None=no goal

        frame_rewards = np.zeros(2, np.float32)
        frame_dones = np.array([is_done, is_done], dtype=bool)
        if is_done and scorer is not None:
            frame_rewards[scorer] = 1.0
            frame_rewards[1 - scorer] = -1.0

        blue_tok = gamestate_to_tokens(state, player_idx=0)
        orange_tok = gamestate_to_tokens(state, player_idx=1)

        frames_tokens.append(np.stack([blue_tok, orange_tok]))
        frames_actions.append(np.stack([team_action[0], team_action[1]]))
        frames_rewards.append(frame_rewards)
        frames_dones.append(frame_dones)

    tokens = np.stack(frames_tokens).astype(np.float32)   # (T, 2, N, F)
    actions = np.stack(frames_actions).astype(np.float32)  # (T, 2, 8)
    rewards = np.stack(frames_rewards).astype(np.float32)  # (T, 2)
    dones = np.stack(frames_dones)                       # (T, 2) bool

    return tokens, actions, rewards, dones


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
                resp = self._session.get(
                    f'{BALLCHASING_API}/replays', params=params)
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
    if manifest_path.exists():
        with open(manifest_path) as f:
            entries = json.load(f)
        return {e['replay_id']: e for e in entries}
    return {}


def _save_manifest(manifest_path: Path, manifest: dict[str, dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(list(manifest.values()), f, indent=2)


def _is_complete(npz_path: Path) -> bool:
    """Return True if npz_path exists and contains all required keys."""
    if not npz_path.exists():
        return False
    try:
        with np.load(npz_path) as data:
            return {'tokens', 'actions', 'rewards', 'dones'}.issubset(data.files)
    except Exception:
        return False


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
    output_dir = Path(output_dir)
    raw_dir = output_dir / 'raw'
    parsed_dir = output_dir / 'parsed'
    manifest_path = output_dir / 'manifest.json'

    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(manifest_path) if resume else {}
    client = BallchasingClient(api_key)

    print(f'Searching ballchasing.com for {count} replays '
          f'(rank: {min_rank}–{max_rank}, playlist: {playlist}) ...')
    replay_list = client.search_replays(count, min_rank, max_rank, playlist)
    print(f'Found {len(replay_list)} replays from API.')

    saved = 0
    skipped = 0
    failed = 0

    for meta in replay_list:
        rid = meta['id']

        if resume and rid in manifest:
            if _is_complete(parsed_dir / f'{rid}.npz'):
                skipped += 1
                continue
            # incomplete file — fall through to re-download

        raw_path = raw_dir / f'{rid}.replay'
        parsed_path = parsed_dir / f'{rid}.npz'

        print(
            f'  [{saved + skipped + 1}/{len(replay_list)}] {rid} ...', end=' ', flush=True)

        try:
            client.download_replay(rid, raw_path)
            tokens, actions, rewards, dones = parse_replay(raw_path)
            np.savez_compressed(parsed_path,
                                tokens=tokens,
                                actions=actions,
                                rewards=rewards,
                                dones=dones)
        except Exception as exc:
            print(f'FAILED ({exc})')
            if raw_path.exists():
                raw_path.unlink()
            failed += 1
            continue

        if not keep_raw:
            raw_path.unlink()

        n_goals = int(dones[:, 0].sum())
        manifest[rid] = {
            'replay_id':     rid,
            'rank':          meta.get('min_rank', {}).get('name', min_rank),
            'playlist':      meta.get('playlist_id', playlist),
            'frame_count':   int(tokens.shape[0]),
            'goal_count':    n_goals,
            'parsed_path':   str(parsed_path.relative_to(output_dir)),
            'downloaded_at': datetime.now(timezone.utc).isoformat(),
        }
        _save_manifest(manifest_path, manifest)

        print(f'OK  ({tokens.shape[0]} frames, {n_goals} goals)')
        saved += 1

    print(f'\nDone. Saved {saved} | Skipped {skipped} | Failed {failed}')
    print(f'Total in manifest: {len(manifest)}')


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Download and parse Rocket League replays.')
    parser.add_argument('--api-key',    default=os.environ.get('BALLCHASING_API_KEY'),
                        help='ballchasing.com API key (or set BALLCHASING_API_KEY env var)')
    parser.add_argument('--count',      type=int, default=10,
                        help='Number of replays to collect (default: 10)')
    parser.add_argument('--output-dir', default='training/replay_data',
                        help='Output directory (default: training/replay_data)')
    parser.add_argument('--min-rank',   default='bronze-1',
                        help='Minimum rank filter (default: bronze-1)')
    parser.add_argument('--max-rank',   default='bronze-3',
                        help='Maximum rank filter (default: bronze-3)')
    parser.add_argument('--playlist',   default='ranked-standard',
                        help='Playlist filter (default: ranked-standard)')
    parser.add_argument('--keep-raw',   action='store_true',
                        help='Keep raw .replay files after parsing')
    parser.add_argument('--no-resume',  action='store_true',
                        help='Re-download replays already in manifest')
    args = parser.parse_args()

    if not args.api_key:
        parser.error(
            '--api-key is required (or set BALLCHASING_API_KEY in your environment)')

    collect(
        api_key=args.api_key,
        count=args.count,
        output_dir=Path(args.output_dir),
        min_rank=args.min_rank,
        max_rank=args.max_rank,
        playlist=args.playlist,
        keep_raw=args.keep_raw,
        resume=not args.no_resume,
    )
