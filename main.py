#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import random
import argparse
from dataclasses import dataclass
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Set

import requests


try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


Coord = Tuple[int, int]


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(p: Coord, w: int, h: int) -> bool:
    return 0 <= p[0] < w and 0 <= p[1] < h


def neighbors4(p: Coord) -> List[Coord]:
    x, y = p
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def tri_points_for_obstacles(k: int) -> int:
    """Очки за k уничтоженных препятствий одной бомбой: 1+2+3+4, максимум 10."""
    m = min(k, 4)
    return (m * (m + 1)) // 2  # 1..m



def show_game_field(arena: Dict[str, Any]):
    """Оригинальная функция пользователя. Оставляю как есть (plt.show блокирует)."""
    size = arena['map_size']
    obstacles = arena['arena']['obstacles']
    if obstacles is not None:
        for i in obstacles:
            plt.plot(i[0], i[1], marker='s', color='green')

    walls = arena['arena']['walls']
    if walls is not None:
        for i in walls:
            plt.plot(i[0], i[1], marker='s', color='grey')

    bombs = arena['arena']['bombs']
    if bombs is not None:
        for i in bombs:
            plt.plot(i['pos'][0], i['pos'][1], marker='o', color='black')
            for j in range(1, i['range'] + 1):
                plt.plot(i['pos'][0] + j, i['pos'][1], marker='*', color='orange')
                plt.plot(i['pos'][0] - j, i['pos'][1], marker='*', color='orange')
                plt.plot(i['pos'][0], i['pos'][1] + j, marker='*', color='orange')
                plt.plot(i['pos'][0], i['pos'][1] - j, marker='*', color='orange')

    bombers = arena['bombers']
    if bombers is not None:
        for i in bombers:
            if i['alive']:
                plt.plot(i['pos'][0], i['pos'][1], marker='o', color='blue')

    enemies = arena['enemies']
    if enemies is not None:
        for i in enemies:
            plt.plot(i['pos'][0], i['pos'][1], marker='o', color='red')

    mobs = arena['mobs']
    if mobs is not None:
        for i in mobs:
            if i['type'] == "ghost":
                plt.plot(i['pos'][0], i['pos'][1], marker='o', color='yellow')
            else:
                plt.plot(i['pos'][0], i['pos'][1], marker='*', color='yellow')
            print(i['type'])

    plt.show()


def show_game_field_live(
    arena: Dict[str, Any],
    plans: Optional[Dict[str, List[Coord]]] = None,
    title: str = "",
):
    if not HAS_PLT:
        return
    plt.clf()
    plt.gca().set_aspect('equal', adjustable='box')

    # Немного сетки
    w, h = arena["map_size"]
    plt.xlim(-1, w)
    plt.ylim(-1, h)
    plt.grid(True, linewidth=0.2, alpha=0.3)

    # Статика + динамика
    obstacles = arena["arena"].get("obstacles") or []
    walls = arena["arena"].get("walls") or []
    bombs = arena["arena"].get("bombs") or []
    bombers = arena.get("bombers") or []
    enemies = arena.get("enemies") or []
    mobs = arena.get("mobs") or []

    for x, y in walls:
        plt.plot(x, y, marker='s', color='grey', markersize=6)
    for x, y in obstacles:
        plt.plot(x, y, marker='s', color='green', markersize=6)

    for b in bombs:
        bx, by = b["pos"]
        plt.plot(bx, by, marker='o', color='black', markersize=6)

    for e in enemies:
        ex, ey = e["pos"]
        plt.plot(ex, ey, marker='o', color='red', markersize=6)

    for m in mobs:
        mx, my = m["pos"]
        if m["type"] == "ghost":
            plt.plot(mx, my, marker='o', color='gold', markersize=6)
        else:
            plt.plot(mx, my, marker='*', color='gold', markersize=8)

    for bo in bombers:
        if bo.get("alive"):
            x, y = bo["pos"]
            plt.plot(x, y, marker='o', color='blue', markersize=7)
            plt.text(x + 0.2, y + 0.2, bo["id"][-4:], fontsize=7, color="blue")

    if plans:
        colors = ["cyan", "magenta", "purple", "dodgerblue", "teal", "brown"]
        for idx, (bid, path) in enumerate(plans.items()):
            if not path:
                continue
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            plt.plot(xs, ys, color=colors[idx % len(colors)], linewidth=1.5, alpha=0.8)

    if title:
        plt.title(title)
    plt.pause(0.001)


# -----------------------------
# Rate limiter: 3 req/sec
# -----------------------------
class RateLimiter:
    def __init__(self, max_per_sec: int = 3):
        self.max_per_sec = max_per_sec
        self.ts: deque[float] = deque()

    def wait(self):
        now = time.monotonic()
        # выкидываем старые
        while self.ts and now - self.ts[0] > 1.0:
            self.ts.popleft()
        if len(self.ts) >= self.max_per_sec:
            sleep_for = 1.0 - (now - self.ts[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.ts.append(time.monotonic())


# -----------------------------
# API client
# -----------------------------
class ApiClient:
    def __init__(self, base_url: str, token: str, timeout: float = 2.5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"X-Auth-Token": token})
        self.rl = RateLimiter(3)

    def get(self, path: str) -> Dict[str, Any]:
        self.rl.wait()
        r = self.s.get(self.base_url + path, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.rl.wait()
        r = self.s.post(self.base_url + path, json=payload, timeout=self.timeout)
        # сервер иногда возвращает gamesdk.PublicError с 200/400 — парсим как JSON
        try:
            data = r.json()
        except Exception:
            r.raise_for_status()
            return {}
        if r.status_code >= 400:
            # не падаем, а просто отдаём ошибку наверх
            return data
        return data

    def arena(self) -> Dict[str, Any]:
        return self.get("/api/arena")

    def boosters(self) -> Dict[str, Any]:
        return self.get("/api/booster")

    def move(self, bombers_cmd: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.post("/api/move", {"bombers": bombers_cmd})

    def buy_booster(self, booster_id: int) -> Dict[str, Any]:
        return self.post("/api/booster", {"booster": booster_id})


# -----------------------------
# Planner
# -----------------------------
@dataclass
class BoosterState:
    points: int = 0
    speed: int = 2
    bomb_range: int = 1
    bomb_delay_ms: int = 8000
    can_pass_bombs: bool = False
    can_pass_obstacles: bool = False
    can_pass_walls: bool = False


@dataclass
class Plan:
    path: List[Coord]
    bombs: List[Coord]
    score: float


class BomberBot:
    """
    Основная идея:
    - Сначала: если мы в опасности (взрыв скоро / моб рядом) -> уходим в безопасную клетку.
    - Иначе: ищем "лучшее место для бомбы" (макс препятствий 3-4 в кресте) в пределах достижимости,
      и обязательно строим путь "дойти -> поставить -> уйти" (в конце стоим вне линии x==bx или y==by).
    - Если бомба недоступна: двигаемся к клетке рядом с препятствием (стейджинг).
    """

    def __init__(self, seed: int = 1):
        self.rng = random.Random(seed)
        self.last_booster_fetch = 0.0
        self.cached_boosters: Optional[Dict[str, Any]] = None

        # Маппинг бустер -> int НЕ описан явно в swagger (там int, а в available type строка).
        # Поэтому делаем "best effort": сначала пробуем этот enum, если 400 — в дальнейшем можно отключить покупки.
        self.booster_enum_guess = {
            "pockets": 0,       # +1 bombs
            "bomb": 1,          # +1 range
            "body": 2,          # +1 speed
            "vision": 3,        # +view
            "soft": 4,          # +bombers
            "armor": 5,         # +armor
            "fuse": 6,          # -delay
            "acrobatics": 7,    # pass...
            # возможны другие имена у сервера; ниже есть fallback по подстрокам
        }
        self.booster_purchase_disabled = False
        self.last_purchase_time = 0.0

    # -------- world parsing --------
    def _sets_from_arena(self, arena: Dict[str, Any]) -> Tuple[Set[Coord], Set[Coord], List[Dict[str, Any]], Set[Coord]]:
        walls = set(map(tuple, arena["arena"].get("walls") or []))
        obstacles = set(map(tuple, arena["arena"].get("obstacles") or []))
        bombs = arena["arena"].get("bombs") or []
        bomb_pos = set(tuple(b["pos"]) for b in bombs)
        return walls, obstacles, bombs, bomb_pos

    def _entities(self, arena: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        bombers = arena.get("bombers") or []
        enemies = arena.get("enemies") or []
        mobs = arena.get("mobs") or []
        return bombers, enemies, mobs

    # -------- explosion / hazards --------
    def blast_cells_for_bomb(
        self,
        pos: Coord,
        rng: int,
        walls: Set[Coord],
        obstacles: Set[Coord],
        bombs_block: Set[Coord],
        w: int,
        h: int
    ) -> Set[Coord]:
        """Клетки креста, учитывая, что луч останавливается на стене/препятствии/бомбе (и включает блокер)."""
        out: Set[Coord] = set()
        out.add(pos)

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for k in range(1, rng + 1):
                p = (pos[0] + dx * k, pos[1] + dy * k)
                if not in_bounds(p, w, h):
                    break
                if p in walls:
                    break
                out.add(p)
                # блокеры: препятствие или бомба
                if p in obstacles or p in bombs_block:
                    break
        return out

    def hazard_time_map(
        self,
        bombs: List[Dict[str, Any]],
        walls: Set[Coord],
        obstacles: Set[Coord],
        w: int,
        h: int
    ) -> Dict[Coord, float]:
        """Для каждой клетки: через сколько секунд её накроет ближайший взрыв (по видимым бомбам)."""
        bombs_block = set(tuple(b["pos"]) for b in bombs)
        hazard: Dict[Coord, float] = {}
        for b in bombs:
            bx, by = b["pos"]
            rng = int(b.get("range", 1))
            timer = float(b.get("timer", 9999.0))
            cells = self.blast_cells_for_bomb((bx, by), rng, walls, obstacles, bombs_block, w, h)
            for c in cells:
                prev = hazard.get(c, float("inf"))
                if timer < prev:
                    hazard[c] = timer
        return hazard

    # -------- movement graph --------
    def is_walkable(
        self,
        p: Coord,
        w: int,
        h: int,
        walls: Set[Coord],
        obstacles: Set[Coord],
        bombs_block: Set[Coord],
        state: BoosterState,
        mob_cells_block: Set[Coord],
    ) -> bool:
        if not in_bounds(p, w, h):
            return False
        if (not state.can_pass_walls) and (p in walls):
            return False
        if (not state.can_pass_obstacles) and (p in obstacles):
            return False
        if (not state.can_pass_bombs) and (p in bombs_block):
            return False
        if p in mob_cells_block:
            return False
        return True

    def safe_bfs(
        self,
        start: Coord,
        w: int,
        h: int,
        walls: Set[Coord],
        obstacles: Set[Coord],
        bombs_block: Set[Coord],
        state: BoosterState,
        hazard: Dict[Coord, float],
        mob_cells_block: Set[Coord],
        max_steps: int,
        start_time: float = 0.0,
        margin: float = 0.20,
    ) -> Tuple[Dict[Coord, int], Dict[Coord, Optional[Coord]]]:
        """
        BFS по клеткам, но отбрасываем переходы, если к моменту прихода клетка уже будет взорвана.
        hazard[c] = seconds_until_explosion_from_now.
        """
        dist: Dict[Coord, int] = {start: 0}
        parent: Dict[Coord, Optional[Coord]] = {start: None}
        q = deque([start])

        speed = max(1, int(state.speed))
        while q:
            v = q.popleft()
            dv = dist[v]
            if dv >= max_steps:
                continue

            for nb in neighbors4(v):
                if nb in dist:
                    continue
                if not self.is_walkable(nb, w, h, walls, obstacles, bombs_block, state, mob_cells_block):
                    continue

                # время прихода
                t_arrive = start_time + (dv + 1) / speed
                t_h = hazard.get(nb, float("inf"))
                if t_arrive >= (t_h - margin):
                    continue

                dist[nb] = dv + 1
                parent[nb] = v
                q.append(nb)

        return dist, parent

    def reconstruct_path(self, parent: Dict[Coord, Optional[Coord]], goal: Coord) -> List[Coord]:
        if goal not in parent:
            return []
        path = []
        cur: Optional[Coord] = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path  # включает старт

    # -------- bomb value --------
    def bomb_value(
        self,
        bomb_pos: Coord,
        bomb_range: int,
        walls: Set[Coord],
        obstacles: Set[Coord],
        bombs_block: Set[Coord],
        enemies: List[Dict[str, Any]],
        mobs: List[Dict[str, Any]],
        w: int,
        h: int
    ) -> Tuple[int, int, int, Set[Coord]]:
        """
        Возвращает:
        - k_obs: сколько препятствий взорвётся (0..4)
        - k_enemy: сколько врагов (видимых и без safe_time) накроет
        - k_mob: сколько мобов (safe_time==0) накроет
        - blast_cells: множество клеток взрыва
        """
        blast = self.blast_cells_for_bomb(bomb_pos, bomb_range, walls, obstacles, bombs_block, w, h)

        # Считаем препятствия: максимум 4, по одному на направление (как в правилах — луч останавливается)
        k_obs = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for k in range(1, bomb_range + 1):
                p = (bomb_pos[0] + dx * k, bomb_pos[1] + dy * k)
                if not in_bounds(p, w, h):
                    break
                if p in walls:
                    break
                if p in bombs_block:
                    break
                if p in obstacles:
                    k_obs += 1
                    break

        k_enemy = 0
        for e in enemies:
            if int(e.get("safe_time", 0)) > 0:
                continue
            ep = tuple(e["pos"])
            if ep in blast:
                k_enemy += 1

        k_mob = 0
        for m in mobs:
            if int(m.get("safe_time", 0)) > 0:
                # спит/неуязвим
                continue
            mp = tuple(m["pos"])
            if mp in blast:
                k_mob += 1

        return k_obs, k_enemy, k_mob, blast

    # -------- escape --------
    def pick_escape_cell(
        self,
        start_after_plant: Coord,
        bomb_pos: Coord,
        explosion_time: float,
        w: int,
        h: int,
        walls: Set[Coord],
        obstacles: Set[Coord],
        bombs_block: Set[Coord],
        state: BoosterState,
        hazard: Dict[Coord, float],
        mob_cells_block: Set[Coord],
        steps_left: int,
        own_blast: Set[Coord],
        start_time: float,
        margin: float = 0.20,
    ) -> Optional[Tuple[Coord, List[Coord]]]:
        """
        Ищем клетку, где:
        - успеваем прийти до explosion_time
        - в момент explosion_time клетка НЕ в own_blast
        - и клетку не накрывает другая бомба к explosion_time
        """
        # Быстрый приоритет: клетки, где x!=bx и y!=by (точно вне линии), в пределах малых шагов
        # Но всё равно учитываем другие бомбы.
        dist, parent = self.safe_bfs(
            start_after_plant, w, h, walls, obstacles, bombs_block, state,
            hazard, mob_cells_block, max_steps=steps_left, start_time=start_time, margin=margin
        )

        speed = max(1, int(state.speed))

        # Соберём кандидатов и выберем лучший по "запасу времени" + отдалённости от мобов
        best: Optional[Tuple[float, Coord]] = None

        # небольшая функция "анти-моб"
        def mob_penalty(c: Coord) -> float:
            # чем ближе к мобу, тем хуже
            # (простая эвристика: min manhattan до моба)
            # mob_cells_block уже исключает ровно клетки мобов, но не соседние
            return 0.0  # оставим минимально, чтобы не переусложнять

        for c, d in dist.items():
            t_arrive = start_time + d / speed
            if t_arrive >= (explosion_time - margin):
                continue
            # в момент взрыва мы стоим на c
            if c in own_blast:
                continue
            if explosion_time >= (hazard.get(c, float("inf")) - margin):
                # другая бомба накроет эту клетку к моменту взрыва нашей/в тот же момент
                continue

            slack = (explosion_time - t_arrive)
            score = slack - 0.01 * d - mob_penalty(c)
            if (best is None) or (score > best[0]):
                best = (score, c)

        if best is None:
            return None

        escape_goal = best[1]
        full = self.reconstruct_path(parent, escape_goal)
        if not full:
            return None
        # full включает start_after_plant; нам нужен хвост (без старта)
        return escape_goal, full[1:]

    # -------- main planning per bomber --------
    def plan_for_bomber(
        self,
        bomber: Dict[str, Any],
        arena: Dict[str, Any],
        state: BoosterState,
        walls: Set[Coord],
        obstacles: Set[Coord],
        bombs: List[Dict[str, Any]],
        bombs_block: Set[Coord],
        hazard: Dict[Coord, float],
        enemies: List[Dict[str, Any]],
        mobs: List[Dict[str, Any]],
        reserved_targets: Set[Coord],
    ) -> Optional[Plan]:
        w, h = arena["map_size"]
        start = tuple(bomber["pos"])
        bombs_available = int(bomber.get("bombs_available", 0))
        armor = int(bomber.get("armor", 0))
        safe_time_ms = int(bomber.get("safe_time", 0))

        # Мобы как "жёсткие" блокеры
        mob_cells_block = set()
        for m in mobs:
            if int(m.get("safe_time", 0)) > 0:
                continue
            mob_cells_block.add(tuple(m["pos"]))

        # 0) Если сейчас стоим в клетке, которую скоро накроет взрыв — уходим немедленно
        danger_t = hazard.get(start, float("inf"))
        if danger_t < 1.2 and safe_time_ms <= 0:
            dist, parent = self.safe_bfs(
                start, w, h, walls, obstacles, bombs_block, state,
                hazard, mob_cells_block, max_steps=12, start_time=0.0
            )
            # выбрать любую клетку без hazard (или с большим временем)
            best = None
            for c, d in dist.items():
                ht = hazard.get(c, float("inf"))
                if ht > 3.0 and c not in mob_cells_block:
                    # предпочитаем ближе
                    score = ht - 0.05 * d
                    if best is None or score > best[0]:
                        best = (score, c)
            if best:
                path_full = self.reconstruct_path(parent, best[1])
                path = path_full[1:][:30]
                return Plan(path=path, bombs=[], score=1.0)

        # 1) Если рядом моб (вплотную) — отходим (моб убивает при пересечении)
        if safe_time_ms <= 0:
            for nb in neighbors4(start) + [start]:
                if nb in mob_cells_block:
                    # отходим на клетку, максимизирующую дистанцию до ближайшего моба
                    dist, parent = self.safe_bfs(
                        start, w, h, walls, obstacles, bombs_block, state,
                        hazard, mob_cells_block, max_steps=10, start_time=0.0
                    )
                    if not dist:
                        break
                    best = None
                    mob_list = list(mob_cells_block)
                    for c, d in dist.items():
                        if not mob_list:
                            md = 999
                        else:
                            md = min(manhattan(c, mc) for mc in mob_list)
                        score = md - 0.03 * d
                        if best is None or score > best[0]:
                            best = (score, c)
                    if best:
                        path_full = self.reconstruct_path(parent, best[1])
                        return Plan(path=path_full[1:][:30], bombs=[], score=2.0)
                    break

        # 2) Пытаемся поставить бомбу с максимальным профитом (3-4 препятствия, либо киллы)
        if bombs_available > 0:
            max_depth_to_target = 12  # чтобы не тратить длинные пути (и быстрее пере-планировать)
            dist, parent = self.safe_bfs(
                start, w, h, walls, obstacles, bombs_block, state,
                hazard, mob_cells_block, max_steps=max_depth_to_target, start_time=0.0
            )

            candidates: List[Tuple[float, Coord, List[Coord], Set[Coord], int, int, int]] = []
            # Рассмотрим только достижимые клетки, где можно ставить бомбу (не стена/препятствие/бомба/моб)
            for c, d in dist.items():
                if c == start:
                    continue
                if c in reserved_targets:
                    continue
                if not self.is_walkable(c, w, h, walls, obstacles, bombs_block, state, mob_cells_block):
                    continue
                # не ставим на клетку бомбы
                if c in bombs_block:
                    continue
                # оценка взрыва
                k_obs, k_enemy, k_mob, blast = self.bomb_value(
                    c, state.bomb_range, walls, obstacles, bombs_block, enemies, mobs, w, h
                )
                if k_obs == 0 and k_enemy == 0 and k_mob == 0:
                    continue

                # Не хотим ставить "пустые" или слабые бомбы, если можно лучше
                pts = tri_points_for_obstacles(k_obs) + 10 * (k_enemy + k_mob)

                # Путь до c
                path_full = self.reconstruct_path(parent, c)
                path_to_c = path_full[1:]
                if not path_to_c:
                    continue
                if len(path_to_c) > 30:
                    continue

                # Тайминг: дойдём, поставим, убежим
                speed = max(1, int(state.speed))
                t_arrive = len(path_to_c) / speed
                bomb_delay = state.bomb_delay_ms / 1000.0
                t_expl = t_arrive + bomb_delay

                # План побега: после постановки мы можем продолжить двигаться (в том же запросе)
                steps_left = 30 - len(path_to_c)
                if steps_left <= 0:
                    continue

                # Модель: после постановки бомбы клетка с бомбой становится блокером (если нельзя проходить сквозь бомбы)
                bombs_block_after = set(bombs_block)
                bombs_block_after.add(c)

                own_blast = self.blast_cells_for_bomb(c, state.bomb_range, walls, obstacles, bombs_block_after, w, h)

                escape = self.pick_escape_cell(
                    start_after_plant=c,
                    bomb_pos=c,
                    explosion_time=t_expl,
                    w=w, h=h,
                    walls=walls, obstacles=obstacles,
                    bombs_block=bombs_block_after,
                    state=state,
                    hazard=hazard,
                    mob_cells_block=mob_cells_block,
                    steps_left=steps_left,
                    own_blast=own_blast,
                    start_time=t_arrive,
                )
                if escape is None:
                    continue
                _, escape_path = escape

                full_path = (path_to_c + escape_path)[:30]
                if c not in full_path:
                    # бомба должна быть "на пути"; если вдруг escape_path пуст, c всё равно в path_to_c
                    continue

                # Финальная точка должна быть безопасной на момент взрыва
                end = full_path[-1] if full_path else start
                if end in own_blast:
                    continue
                if t_expl >= (hazard.get(end, float("inf")) - 0.20) and safe_time_ms <= 0:
                    continue

                # Счёт: хотим больше очков при меньшем пути и меньшем риске
                # бонус за 4 препятствия (макс 10)
                bonus4 = 2.0 if k_obs >= 4 else 0.0
                score = (pts * 5.0) + bonus4 - (0.25 * len(full_path))

                # Чуть штрафуем за "килл-охоту", если нет препятствий (менее стабильна)
                if k_obs == 0 and (k_enemy + k_mob) > 0:
                    score -= 1.0

                candidates.append((score, c, full_path, own_blast, k_obs, k_enemy, k_mob))

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                score, bomb_cell, full_path, _, k_obs, k_enemy, k_mob = candidates[0]
                # важная деталь: bombs список может быть в произвольном порядке, но координата должна встретиться в path
                bombs_list = [bomb_cell]
                reserved_targets.add(bomb_cell)
                return Plan(path=full_path, bombs=bombs_list, score=score)

        # 3) Нет хорошей бомбы / нет бомб: идём к клетке рядом с препятствием (подготовка)
        # Выбираем ближайшую клетку, рядом с которой есть препятствие.
        max_depth = 14
        dist, parent = self.safe_bfs(
            start, w, h, walls, obstacles, bombs_block, state,
            hazard, mob_cells_block, max_steps=max_depth, start_time=0.0
        )
        best = None
        for c, d in dist.items():
            if c == start:
                continue
            # клетка должна быть ходибельной
            if not self.is_walkable(c, w, h, walls, obstacles, bombs_block, state, mob_cells_block):
                continue
            # рядом должно быть препятствие (чтобы следующей командой поставить бомбу выгоднее)
            near_obs = 0
            for nb in neighbors4(c):
                if nb in obstacles:
                    near_obs += 1
            if near_obs == 0:
                continue
            # ещё: хотим не стоять на "линии" ближайшей бомбы (из hazard) если таймер небольшой
            ht = hazard.get(c, float("inf"))
            score = (near_obs * 2.0) - 0.10 * d
            if ht < 2.5:
                score -= 5.0
            if best is None or score > best[0]:
                best = (score, c)

        if best:
            path_full = self.reconstruct_path(parent, best[1])
            return Plan(path=path_full[1:][:30], bombs=[], score=0.5)

        # 4) Если вообще нечего делать — случайный безопасный шаг (разведка)
        # (карта частично скрыта: неизвестные клетки считаем свободными)
        safe_steps = []
        for nb in neighbors4(start):
            if self.is_walkable(nb, w, h, walls, obstacles, bombs_block, state, mob_cells_block):
                if (0.5 / max(1, state.speed)) < hazard.get(nb, float("inf")):
                    safe_steps.append(nb)
        if safe_steps:
            self.rng.shuffle(safe_steps)
            return Plan(path=[safe_steps[0]], bombs=[], score=0.1)

        return None

    # -------- boosters (best effort) --------
    def parse_booster_state(self, boosters_json: Dict[str, Any]) -> BoosterState:
        st = boosters_json.get("state") or {}
        return BoosterState(
            points=int(st.get("points", 0)),
            speed=int(st.get("speed", 2)),
            bomb_range=int(st.get("bomb_range", 1)),
            bomb_delay_ms=int(st.get("bomb_delay", 8000)),
            can_pass_bombs=bool(st.get("can_pass_bombs", False)),
            can_pass_obstacles=bool(st.get("can_pass_obstacles", False)),
            can_pass_walls=bool(st.get("can_pass_walls", False)),
        )

    def choose_booster_to_buy(self, boosters_json: Dict[str, Any]) -> Optional[str]:
        """
        Выбор по типу (строка из available.type).
        Приоритет под скоринг препятствий:
        1) bombs (pockets) -> больше одновременных бомб
        2) range (bomb) -> чаще 3-4 препятствия
        3) fuse (bomb_delay) -> чаще циклы, но риск выше
        4) speed (body) -> быстрее добираться/уходить
        5) armor -> страховка
        """
        if self.booster_purchase_disabled:
            return None

        avail = boosters_json.get("available") or []
        st = self.parse_booster_state(boosters_json)

        if st.points <= 0 or not avail:
            return None

        # нормализуем имена
        def norm(s: str) -> str:
            return (s or "").strip().lower()

        types = [norm(x.get("type", "")) for x in avail]

        # эвристика по подстрокам (на случай других названий)
        def has(substr: str) -> Optional[str]:
            for t in types:
                if substr in t:
                    return t
            return None

        # приоритеты
        for want in [
            ("bombs", ["pocket", "bombs", "pockets"]),
            ("range", ["range", "bomb_range", "bomb"]),
            ("delay", ["delay", "fuse", "bomb_delay"]),
            ("speed", ["speed", "body"]),
            ("armor", ["armor"]),
            ("view", ["view", "vision"]),
            ("bombers", ["bomber", "soft"]),
            ("acrobatics", ["acro", "pass"]),
        ]:
            _, keys = want
            for k in keys:
                t = has(k)
                if t is None:
                    continue
                # проверим, что хватает очков
                # в available есть cost
                for x in avail:
                    if norm(x.get("type", "")) == t:
                        cost = int(x.get("cost", 999999))
                        if cost <= st.points:
                            return t
        return None

    def booster_type_to_id_guess(self, booster_type: str) -> Optional[int]:
        t = (booster_type or "").lower()
        # сначала прямое
        if t in self.booster_enum_guess:
            return self.booster_enum_guess[t]
        # по подстрокам
        if "pocket" in t or "bombs" in t:
            return self.booster_enum_guess.get("pockets")
        if "range" in t:
            return self.booster_enum_guess.get("bomb")
        if "speed" in t:
            return self.booster_enum_guess.get("body")
        if "vision" in t or "view" in t:
            return self.booster_enum_guess.get("vision")
        if "armor" in t:
            return self.booster_enum_guess.get("armor")
        if "delay" in t or "fuse" in t:
            return self.booster_enum_guess.get("fuse")
        if "acro" in t or "pass" in t:
            return self.booster_enum_guess.get("acrobatics")
        if "bomber" in t or "soft" in t:
            return self.booster_enum_guess.get("soft")
        return None

    # -------- tick --------
    def tick(
        self,
        api: ApiClient,
        enable_viz: bool = False,
    ):
        arena = api.arena()
        if arena.get("errors"):
            # не валимся, но полезно видеть
            pass

        w, h = arena["map_size"]
        walls, obstacles, bombs, bombs_block = self._sets_from_arena(arena)
        bombers, enemies, mobs = self._entities(arena)

        # бустеры иногда обновляем (редко, чтобы не съедать лимит запросов)
        boosters_json = None
        now = time.monotonic()
        if (now - self.last_booster_fetch) > 2.0:
            try:
                boosters_json = api.boosters()
                self.cached_boosters = boosters_json
                self.last_booster_fetch = now
            except Exception:
                boosters_json = self.cached_boosters
        else:
            boosters_json = self.cached_boosters

        state = BoosterState()
        if boosters_json:
            state = self.parse_booster_state(boosters_json)

        hazard = self.hazard_time_map(bombs, walls, obstacles, w, h)

        # планируем для каждого бомбера
        reserved_targets: Set[Coord] = set()
        bomber_cmds: List[Dict[str, Any]] = []
        plans_for_viz: Dict[str, List[Coord]] = {}

        # сортируем: сначала те, у кого есть бомбы и кто ближе к центру (чтобы быстрее начать фарм)
        def sort_key(b: Dict[str, Any]) -> Tuple[int, int]:
            ba = int(b.get("bombs_available", 0))
            pos = tuple(b.get("pos", (0, 0)))
            center = (w // 2, h // 2)
            return (-ba, manhattan(pos, center))

        for b in sorted(bombers, key=sort_key):
            if not b.get("alive", False):
                continue
            if not b.get("can_move", False):
                continue

            plan = self.plan_for_bomber(
                bomber=b, arena=arena, state=state,
                walls=walls, obstacles=obstacles,
                bombs=bombs, bombs_block=bombs_block,
                hazard=hazard, enemies=enemies, mobs=mobs,
                reserved_targets=reserved_targets
            )
            if plan is None:
                continue

            # API ожидает список координат
            path_payload = [[int(x), int(y)] for (x, y) in plan.path][:30]
            bombs_payload = [[int(x), int(y)] for (x, y) in plan.bombs]

            bomber_cmds.append({
                "id": b["id"],
                "path": path_payload,
                "bombs": bombs_payload
            })
            plans_for_viz[b["id"]] = plan.path

        # покупка бустера: best effort (и очень редко, чтобы не съесть лимит)
        # Если сервер возвращает 400 — отключаем покупки (чтобы не тратить запросы).
        if boosters_json and (not self.booster_purchase_disabled):
            if (now - self.last_purchase_time) > 2.5:
                bt = self.choose_booster_to_buy(boosters_json)
                if bt is not None:
                    bid = self.booster_type_to_id_guess(bt)
                    if bid is not None:
                        resp = api.buy_booster(bid)
                        self.last_purchase_time = now
                        if resp and int(resp.get("code", 0)) != 0 and resp.get("errors"):
                            # Скорее всего неверный id enum; чтобы не спамить 400 — отключаем покупки
                            self.booster_purchase_disabled = True

        # отправляем ходы
        if bomber_cmds:
            api.move(bomber_cmds)

        if enable_viz and HAS_PLT:
            show_game_field_live(
                arena,
                plans=plans_for_viz,
                title=f"round={arena.get('round','')} score={arena.get('raw_score','?')} speed={state.speed} range={state.bomb_range}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "https://games-test.datsteam.dev"))
    ap.add_argument("--token", default=os.environ.get("X_AUTH_TOKEN") or os.environ.get("DATS_TOKEN") or "")
    ap.add_argument("--viz", action="store_true", help="live visualization with matplotlib")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if not args.token:
        raise SystemExit("Нет токена. Укажите --token или переменную окружения X_AUTH_TOKEN.")

    api = ApiClient(base_url=args.base_url, token=args.token)

    bot = BomberBot(seed=args.seed)

    if args.viz and HAS_PLT:
        plt.ion()
        plt.figure(figsize=(7, 7))

    # Основной цикл
    while True:
        try:
            bot.tick(api, enable_viz=bool(args.viz and HAS_PLT))
        except requests.RequestException:
            # сеть/сервер: чуть подождём и продолжим
            time.sleep(0.3)
        except KeyboardInterrupt:
            break
        except Exception:
            # чтобы бот не падал из-за единичной неожиданности
            time.sleep(0.1)


if __name__ == "__main__":
    main()
