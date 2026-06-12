# Apartment Map — Prolog Final Project

A knowledge-based model of an apartment built in **SWI-Prolog**, with a
**Python + Tkinter** GUI driving the Prolog engine through `pyswip`.

Final project for the **Knowledge Engineering Methods (MIW)** course
at the **Polish-Japanese Academy of Information Technology (PJATK)**.

- Graded **18/18**

---

## Overview

The apartment is modelled as a Prolog knowledge base — rooms, furniture,
spatial relations, and an agent that navigates the space. A Tkinter GUI
sits on top: it renders the map, animates the agent walking room-to-room,
and lets the user query the knowledge base interactively.

- 5 rooms (living room, bedroom, kitchen, bathroom, corridor)
- 16 furniture items with spatial relations between them
- Dynamic add / move / remove of objects at runtime
- Shortest-path navigation via recursive Prolog rules
- Free-form Prolog query console embedded in the GUI

---

## Screenshots

**Navigate tab** — quick-access buttons for common targets and a dropdown for any object in the knowledge base.

<img width="1198" height="707" alt="SCR-20260608-blco" src="https://github.com/user-attachments/assets/72bdd620-b2fe-4585-995e-3b740063256f" />

**Modify tab** — add, move, or remove objects at runtime. The underlying Prolog facts are updated via `assertz` / `retract`.

<img width="1197" height="701" alt="SCR-20260608-bleo" src="https://github.com/user-attachments/assets/89f556bc-e29b-4cb8-ab58-3e377cb8909c" />

**Query tab** — free-form Prolog console with one-click example queries.

<img width="1197" height="705" alt="SCR-20260608-blgc" src="https://github.com/user-attachments/assets/a7bce9b5-e4f9-4324-8e23-1e2d0bb6032d" />

---

## Topics Covered

- **Knowledge representation** — facts, compound terms, and rules describing the apartment
- **Spatial reasoning** — `on`, `next_to`, `between`, `same_room`, adjacency, transitive closure
- **Dynamic knowledge bases** — `assertz` / `retract` for runtime fact modification
- **Recursive search** — pathfinding with backtracking and cycle prevention
- **List processing** — `findall`, `member`, `length`, `reverse` for collection-style queries
- **Language interoperability** — Python (Tkinter GUI) driving SWI-Prolog via `pyswip`

---

## Requirements Coverage

| Requirement                       | Implementation                                                                                  |
| --------------------------------- | ----------------------------------------------------------------------------------------------- |
| Terms / compound terms (facts)    | `room/1`, `object/2`, `door/2`, `on/2`, `next_to/2`                                             |
| Clauses (new concepts)            | `adjacent_room/2`, `neighbor/2`, `between/3`, `same_room/2`, `empty_room/1`                     |
| Dynamic elements                  | `add_object/2`, `remove_object/2`, `move_object/3` — built on `assertz` / `retract`             |
| Recursion                         | `on_transitive/2`, `path_helper/4`, `walk/1`, `shortest_in_list/2`                              |
| Unification and cut               | Shared variables (e.g. `Room` in `same_room/2`); cut in `add_object`, `walk`, `path_helper`     |
| Lists                             | `findall/3`, `length/2`, `member/2`, `reverse/2` in `objects_in_room`, `shortest_path`, `where_is` |
| Spatial relations                 | `between/3`, `next_to/2`, `on/2`, `on_transitive/2`, `same_room/2`                              |
| Specific navigation task          | `go_near/1`, `go_to_tv/0` — animated walk along the shortest path                               |
| **Bonus: language integration**   | Python (Tkinter) ↔ `pyswip` ↔ SWI-Prolog                                                        |

---

## Project

### Prolog Backend — `apartment.pl`

The knowledge base and reasoning rules:

- **Facts** describe rooms, doors, objects per room, and `on` / `next_to` relations
- **Rules** derive higher-level relations: symmetric adjacency, transitive `on`, betweenness, same-room membership
- **Pathfinding** uses recursive depth-first search with a visited list to prevent cycles; `shortest_path/3` collects all paths via `findall` and picks the shortest by length
- **Dynamic operations** (`add_object`, `remove_object`, `move_object`) mutate the knowledge base at runtime; cut prevents the error fallback from firing on success
- **Agent location** is itself a dynamic fact (`location(me, Room)`) updated step-by-step as the agent walks

### Python GUI — `apartment_gui.py`

A Tkinter front-end that loads `apartment.pl` through `pyswip` and exposes the knowledge base visually:

- **Map canvas** — rooms drawn as rectangles with door connections; the agent is rendered as a coloured marker that animates between rooms
- **Navigate tab** — one-click buttons for common targets (TV, bed, fridge…) plus a dropdown for any object in the KB
- **Modify tab** — add, move, or remove objects; the underlying Prolog facts are updated via `assertz` / `retract`
- **Query tab** — free-form Prolog console with one-click example queries
- **Activity log** — every interaction with the Prolog engine is echoed for inspection

---

## Running

### Prolog only

```bash
swipl apartment.pl
?- help.
?- go_to_tv.
?- shortest_path(bathroom, kitchen, P).
?- objects_in_room(living_room, L).
```

### With the GUI

SWI-Prolog must be installed system-wide. Then:

```bash
pip install pyswip
python apartment_gui.py
```

`apartment.pl` and `apartment_gui.py` must live in the same directory —
the GUI loads the Prolog file relative to its own location.

---

## Project Structure

```
apartment.pl       — Prolog knowledge base + rules
apartment_gui.py   — Tkinter GUI driving the Prolog engine via pyswip
screenshots/       — GUI screenshots used in this README
README.md
```

---

## Stack

SWI-Prolog · Python · Tkinter · pyswip

---

## Notes

This is my own solution to the final project — shared as study material
and a reference implementation, not for direct submission.
