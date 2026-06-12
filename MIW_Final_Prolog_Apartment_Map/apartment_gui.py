"""
Apartment Prolog GUI
====================
A tkinter GUI that uses pyswip to interact with the apartment.pl Prolog backend.

Features:
- Visual apartment map with rooms, objects, and an animated agent
- Quick navigation buttons (Go to TV, Go to Bed, etc.)
- Dynamic add/move/remove of objects
- Free-form Prolog query console
- Activity log

Run:
    python apartment_gui.py
"""

import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from pyswip import Prolog


class ApartmentApp:
    # Visual layout of rooms on the canvas (x1, y1, x2, y2, label)
    ROOMS = {
        'bedroom':     (220, 20, 440, 150, "BEDROOM"),
        'bathroom':    (20, 190, 200, 320, "BATHROOM"),
        'corridor':    (220, 190, 440, 320, "CORRIDOR"),
        'living_room': (460, 190, 660, 320, "LIVING ROOM"),
        'kitchen':     (460, 360, 660, 490, "KITCHEN"),
    }

    # Door connections (drawn as dashed lines between room centers)
    DOORS = [
        ('bedroom', 'corridor'),
        ('bathroom', 'corridor'),
        ('living_room', 'corridor'),
        ('living_room', 'kitchen'),
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("Apartment Prolog Navigator")
        self.root.geometry("1200x680")

        # Initialize Prolog and load the apartment knowledge base
        self.prolog = Prolog()
        pl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apartment.pl")
        self.prolog.consult(pl_path)

        self.animating = False

        self._build_ui()
        self.refresh()
        self.log("Apartment loaded. Starting position: corridor.")

    # ------------------------------------------------------------------
    # UI CONSTRUCTION
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Main horizontal split: canvas (left) | controls (right)
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ---- LEFT: title + canvas + status ----
        left = ttk.Frame(main)
        ttk.Label(left, text="Apartment Map", font=("Arial", 14, "bold")).pack(pady=4)
        self.canvas = tk.Canvas(
            left, width=700, height=520, bg="white",
            highlightthickness=1, highlightbackground="black"
        )
        self.canvas.pack()
        self.status_var = tk.StringVar(value="Current location: corridor")
        ttk.Label(left, textvariable=self.status_var,
                  font=("Arial", 11, "italic"), foreground="darkblue").pack(pady=4)
        main.add(left, weight=2)

        # ---- RIGHT: tabbed controls + log ----
        right = ttk.Frame(main)
        notebook = ttk.Notebook(right)
        notebook.pack(fill=tk.BOTH, expand=True)

        nav = ttk.Frame(notebook, padding=8)
        notebook.add(nav, text="Navigate")
        self._build_nav_tab(nav)

        mod = ttk.Frame(notebook, padding=8)
        notebook.add(mod, text="Modify")
        self._build_mod_tab(mod)

        qry = ttk.Frame(notebook, padding=8)
        notebook.add(qry, text="Query")
        self._build_qry_tab(qry)

        # Activity log
        ttk.Label(right, text="Activity Log:",
                  font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
        self.log_widget = scrolledtext.ScrolledText(
            right, height=12, wrap=tk.WORD, font=("Courier", 9)
        )
        self.log_widget.pack(fill=tk.BOTH, expand=True)

        main.add(right, weight=1)

    def _build_nav_tab(self, frame):
        ttk.Label(frame, text="Quick navigation:",
                  font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=4)

        quick = [
            ("Go to TV",     "tv"),
            ("Go to Bed",    "bed"),
            ("Go to Fridge", "fridge"),
            ("Go to Shower", "shower"),
            ("Go to Desk",   "desk"),
        ]
        for label, target in quick:
            ttk.Button(frame, text=label,
                       command=lambda t=target: self.go_near(t)).pack(fill=tk.X, pady=2)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(frame, text="Go to any object:",
                  font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=4)
        self.go_combo = ttk.Combobox(frame, values=self._all_objects(), state="readonly")
        self.go_combo.pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Go", command=self._go_combo_handler).pack(fill=tk.X, pady=2)

    def _build_mod_tab(self, frame):
        # Add
        ttk.Label(frame, text="Add object:",
                  font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=4)
        af = ttk.Frame(frame)
        af.pack(fill=tk.X)
        ttk.Label(af, text="Name:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.add_name = ttk.Entry(af)
        self.add_name.grid(row=0, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Label(af, text="Room:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.add_room = ttk.Combobox(af, values=list(self.ROOMS.keys()), state="readonly")
        self.add_room.grid(row=1, column=1, sticky=tk.EW, padx=4, pady=2)
        af.columnconfigure(1, weight=1)
        ttk.Button(frame, text="Add Object", command=self.add_object).pack(fill=tk.X, pady=4)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Move
        ttk.Label(frame, text="Move object:",
                  font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=4)
        mf = ttk.Frame(frame)
        mf.pack(fill=tk.X)
        ttk.Label(mf, text="Object:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.move_obj = ttk.Combobox(mf, values=self._all_objects(), state="readonly")
        self.move_obj.grid(row=0, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Label(mf, text="To room:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.move_to = ttk.Combobox(mf, values=list(self.ROOMS.keys()), state="readonly")
        self.move_to.grid(row=1, column=1, sticky=tk.EW, padx=4, pady=2)
        mf.columnconfigure(1, weight=1)
        ttk.Button(frame, text="Move Object", command=self.move_object).pack(fill=tk.X, pady=4)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Remove
        ttk.Label(frame, text="Remove object:",
                  font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=4)
        self.remove_obj = ttk.Combobox(frame, values=self._all_objects(), state="readonly")
        self.remove_obj.pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Remove Object",
                   command=self.remove_object).pack(fill=tk.X, pady=2)

    def _build_qry_tab(self, frame):
        ttk.Label(frame, text="Enter a Prolog query:",
                  font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=4)
        ttk.Label(frame, text="(without trailing period)",
                  font=("Arial", 9, "italic"), foreground="gray").pack(anchor=tk.W)
        self.query_entry = ttk.Entry(frame)
        self.query_entry.pack(fill=tk.X, pady=4)
        self.query_entry.bind("<Return>", lambda e: self.execute_query())
        ttk.Button(frame, text="Execute",
                   command=self.execute_query).pack(fill=tk.X, pady=2)

        ttk.Label(frame, text="Example queries (click to fill):",
                  font=("Arial", 9, "italic")).pack(anchor=tk.W, pady=(12, 2))
        examples = [
            "neighbor(tv, X)",
            "between(sofa, coffee_table, armchair)",
            "shortest_path(bathroom, kitchen, P)",
            "objects_in_room(living_room, L)",
            "same_room(tv, sofa)",
            "where_is(bed, R)",
            "empty_room(R)",
        ]
        for ex in examples:
            ttk.Button(frame, text=ex,
                       command=lambda q=ex: self._fill_query(q)).pack(fill=tk.X, pady=1)

    # ------------------------------------------------------------------
    # PROLOG QUERY HELPERS
    # ------------------------------------------------------------------
    def _all_objects(self):
        results = list(self.prolog.query("object(O, _)"))
        return sorted({str(r['O']) for r in results})

    def _current_room(self):
        return str(next(self.prolog.query("current_room(R)"))['R'])

    def _objects_in_room(self, room):
        result = next(self.prolog.query(f"objects_in_room({room}, L)"))
        return [str(o) for o in result['L']]

    # ------------------------------------------------------------------
    # ACTIONS
    # ------------------------------------------------------------------
    def go_near(self, obj):
        if self.animating:
            self.log("[!] Already navigating, please wait...")
            return
        try:
            current = self._current_room()
            target = list(self.prolog.query(f"object({obj}, R)"))
            if not target:
                self.log(f"[ERROR] Object '{obj}' not found on the map.")
                return
            target_room = str(target[0]['R'])

            if current == target_room:
                self.log(f">> Already in same room as '{obj}' ({current}).")
                return

            path_result = list(
                self.prolog.query(f"shortest_path({current}, {target_room}, P)")
            )
            if not path_result:
                self.log(f"[ERROR] No path from {current} to {target_room}.")
                return
            path = [str(p) for p in path_result[0]['P']]

            self.log(f">> Going to '{obj}': {' -> '.join(path)}")
            self._animate(path, obj)
        except Exception as e:
            self.log(f"[ERROR] {e}")

    def _animate(self, path, target):
        self.animating = True

        def step(i):
            if i >= len(path):
                self.animating = False
                self.log(f">> Arrived! Now near '{target}'.")
                return
            current = self._current_room()
            list(self.prolog.query(f"retract(location(me, {current}))"))
            list(self.prolog.query(f"assertz(location(me, {path[i]}))"))
            self.log(f"   {current} --> {path[i]}")
            self.refresh()
            self.root.after(700, lambda: step(i + 1))

        step(1)  # start from index 1; index 0 is current room

    def add_object(self):
        name = self.add_name.get().strip().lower().replace(" ", "_")
        room = self.add_room.get()
        if not name or not room:
            messagebox.showwarning("Missing input", "Name and room are required.")
            return
        try:
            list(self.prolog.query(f"add_object({name}, {room})"))
            existing = list(self.prolog.query(f"object({name}, {room})"))
            if existing:
                self.log(f"[+] Added '{name}' to '{room}'.")
                self.add_name.delete(0, tk.END)
                self._refresh_combos()
                self.refresh()
            else:
                self.log(f"[!] Could not add '{name}' to '{room}'.")
        except Exception as e:
            self.log(f"[ERROR] {e}")

    def move_object(self):
        obj = self.move_obj.get()
        to_room = self.move_to.get()
        if not obj or not to_room:
            messagebox.showwarning("Missing input", "Object and target room required.")
            return
        try:
            from_results = list(self.prolog.query(f"object({obj}, R)"))
            if not from_results:
                self.log(f"[!] '{obj}' not found.")
                return
            from_room = str(from_results[0]['R'])
            if from_room == to_room:
                self.log(f"[!] '{obj}' is already in '{to_room}'.")
                return
            list(self.prolog.query(f"move_object({obj}, {from_room}, {to_room})"))
            self.log(f"[~] Moved '{obj}' from '{from_room}' to '{to_room}'.")
            self.refresh()
        except Exception as e:
            self.log(f"[ERROR] {e}")

    def remove_object(self):
        obj = self.remove_obj.get()
        if not obj:
            return
        try:
            results = list(self.prolog.query(f"object({obj}, R)"))
            if not results:
                self.log(f"[!] '{obj}' not found.")
                return
            room = str(results[0]['R'])
            list(self.prolog.query(f"remove_object({obj}, {room})"))
            self.log(f"[-] Removed '{obj}' from '{room}'.")
            self._refresh_combos()
            self.refresh()
        except Exception as e:
            self.log(f"[ERROR] {e}")

    def _go_combo_handler(self):
        obj = self.go_combo.get()
        if obj:
            self.go_near(obj)

    def execute_query(self):
        q = self.query_entry.get().strip()
        if not q:
            return
        try:
            results = list(self.prolog.query(q))
            self.log(f"?- {q}.")
            if not results:
                self.log("   false.")
            else:
                for r in results[:8]:
                    if not r:
                        self.log("   true.")
                        break
                    parts = [f"{k} = {self._fmt(v)}" for k, v in r.items()]
                    self.log("   " + ", ".join(parts))
                if len(results) > 8:
                    self.log(f"   ... ({len(results) - 8} more solutions)")
        except Exception as e:
            self.log(f"[ERROR] {e}")

    def _fmt(self, v):
        if isinstance(v, list):
            return "[" + ", ".join(str(x) for x in v) + "]"
        return str(v)

    def _fill_query(self, q):
        self.query_entry.delete(0, tk.END)
        self.query_entry.insert(0, q)

    def _refresh_combos(self):
        objects = self._all_objects()
        self.go_combo['values'] = objects
        self.move_obj['values'] = objects
        self.remove_obj['values'] = objects

    # ------------------------------------------------------------------
    # DRAWING
    # ------------------------------------------------------------------
    def refresh(self):
        self.draw_map()
        self.status_var.set(f"Current location: {self._current_room()}")

    def draw_map(self):
        c = self.canvas
        c.delete("all")
        current_room = self._current_room()

        # Draw connections first (under rooms)
        for r1, r2 in self.DOORS:
            x1, y1, x2, y2, _ = self.ROOMS[r1]
            cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
            x1b, y1b, x2b, y2b, _ = self.ROOMS[r2]
            cx2, cy2 = (x1b + x2b) / 2, (y1b + y2b) / 2
            c.create_line(cx1, cy1, cx2, cy2,
                          fill="#888", width=3, dash=(6, 4))

        # Draw rooms
        for rname, (x1, y1, x2, y2, label) in self.ROOMS.items():
            is_current = (rname == current_room)
            fill_color = "#FFD27A" if is_current else "#F0F0F0"
            outline_color = "#D2691E" if is_current else "#333"
            outline_width = 3 if is_current else 2
            c.create_rectangle(x1, y1, x2, y2,
                               fill=fill_color, outline=outline_color, width=outline_width)
            c.create_text((x1 + x2) / 2, y1 + 14, text=label,
                          font=("Arial", 11, "bold"), fill="#222")

            # Draw objects in the room
            objects = self._objects_in_room(rname)
            for i, obj in enumerate(objects):
                col = i % 2
                row = i // 2
                ox = x1 + 12 + col * ((x2 - x1 - 24) / 2)
                oy = y1 + 36 + row * 16
                c.create_text(ox, oy, text=f"• {obj}",
                              anchor=tk.W, font=("Arial", 8), fill="#222")

        # Draw the agent (red circle) in the current room
        x1, y1, x2, y2, _ = self.ROOMS[current_room]
        ax, ay = x2 - 22, y2 - 22
        c.create_oval(ax - 12, ay - 12, ax + 12, ay + 12,
                      fill="#E74C3C", outline="#922B21", width=2)
        c.create_text(ax, ay, text="ME",
                      font=("Arial", 7, "bold"), fill="white")

    # ------------------------------------------------------------------
    # LOG
    # ------------------------------------------------------------------
    def log(self, msg):
        self.log_widget.insert(tk.END, msg + "\n")
        self.log_widget.see(tk.END)


def main():
    root = tk.Tk()
    ApartmentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
