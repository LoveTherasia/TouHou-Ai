"""Optional real-game safety/aim layer; does not modify learned weights.

Tracks bullet array slot identities across observations and scores short linear
trajectories. Predictions are deliberately short; lasers are not represented.
"""
from __future__ import annotations

import numpy as np

from ..observation import require_entities


class TacticalGuard:
    directions = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0),
                  (-1, -1), (1, -1), (-1, 1), (1, 1)]
    names = ["", "up", "down", "left", "right", "up_left", "up_right", "down_left", "down_right"]

    def __init__(self):
        self.previous = {}
        self.frame = None
        self.last_direction = 0

    def choose(self, state, frame, learned_action):
        require_entities(state)
        p = np.array([state.player_x, state.player_y])
        bullets = np.asarray(state.bullets, dtype=float).reshape(-1, 2)
        slots = state.raw.get("entities", {}).get("bullets", {}).get("active_slots", [])
        dt = frame-self.frame if self.frame is not None else 0
        velocities = np.zeros_like(bullets)
        if 0 < dt <= 120:
            for i, slot in enumerate(slots):
                if slot in self.previous:
                    v = (bullets[i]-self.previous[slot])/dt
                    if np.linalg.norm(v) < 12:
                        velocities[i] = v
        self.previous = {slot: bullets[i].copy() for i, slot in enumerate(slots)}
        self.frame = frame
        # Aim from below; do not chase enemies into the top of the field.
        enemies = [(x,y) for x,y in state.enemy_positions if 40 < x < 344 and 0 < y < 310]
        aim = min(enemies, key=lambda xy: abs(xy[0]-p[0]))[0] if enemies else 192
        target = np.array([np.clip(aim, 65, 319), 395.0])
        times = np.array([0, 4, 8, 12, 18, 25], dtype=float)
        scores, clearances = [], []
        for index, direction in enumerate(self.directions):
            d = np.array(direction, dtype=float)
            d /= max(1, np.linalg.norm(d))
            path = p + times[:, None]*d*2.0
            path = np.clip(path, [8, 32], [376, 432])
            clearance = 9999.0
            danger = 0.0
            if len(bullets):
                predicted = bullets[None,:,:] + times[:,None,None]*velocities[None,:,:]
                distances = np.linalg.norm(predicted-path[:,None,:], axis=2)
                clearance = float(distances.min())
                danger = float(np.exp(-distances**2/(2*18**2)).sum(axis=1).max())*180
                danger += max(0, 13-clearance)*90
            # Enemy contact uses a larger buffer than small bullet hitboxes.
            if state.enemy_positions:
                ed = np.linalg.norm(np.array(state.enemy_positions)[None,:,:]-path[:,None,:], axis=2).min()
                danger += max(0, 30-ed)*60
            cost = danger + np.linalg.norm(path[-1]-target)*0.12
            cost += max(0, 55-path[-1,0])*0.4 + max(0, path[-1,0]-329)*0.4
            cost += 1.0 if index != self.last_direction else 0
            if (self.names[index] or "shoot") in learned_action:
                cost -= 0.4
            scores.append(cost)
            clearances.append(clearance)
        chosen = int(np.argmin(scores))
        self.last_direction = chosen
        action = (self.names[chosen]+"+" if chosen else "")+"shoot+focus"
        return action, {"predicted_clearance": clearances[chosen], "choice_cost": scores[chosen],
                        "aim_x": float(aim), "frame": frame}
