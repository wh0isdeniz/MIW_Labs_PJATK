% =====================================================
% APARTMENT MAP - PROLOG FINAL PROJECT
% =====================================================
% This project models an apartment with its rooms,
% furniture, spatial relations, and navigation rules.
%

% =====================================================

% Dynamic predicates - modifiable at runtime
:- dynamic(object/2).
:- dynamic(on/2).
:- dynamic(location/2).

% =====================================================
% 1. ROOMS AND CONNECTIONS (BASIC FACTS)
% =====================================================

room(living_room).
room(bedroom).
room(kitchen).
room(bathroom).
room(corridor).

% Door connections (directed)
door(corridor, living_room).
door(corridor, bedroom).
door(corridor, bathroom).
door(living_room, kitchen).

% Adjacency is symmetric
adjacent_room(X, Y) :- door(X, Y).
adjacent_room(X, Y) :- door(Y, X).

% =====================================================
% 2. OBJECTS AND THEIR ROOMS (compound terms)
% =====================================================

% Living room
object(tv, living_room).
object(sofa, living_room).
object(coffee_table, living_room).
object(bookshelf, living_room).
object(armchair, living_room).

% Bedroom
object(bed, bedroom).
object(wardrobe, bedroom).
object(desk, bedroom).
object(chair, bedroom).

% Kitchen
object(fridge, kitchen).
object(stove, kitchen).
object(kitchen_table, kitchen).
object(dishwasher, kitchen).

% Bathroom
object(toilet, bathroom).
object(shower, bathroom).
object(sink, bathroom).

% =====================================================
% 3. ON RELATIONS (objects placed on surfaces)
% =====================================================

on(remote_control, coffee_table).
on(book, coffee_table).
on(lamp, coffee_table).
on(pillow, sofa).
on(blanket, armchair).
on(plant, bookshelf).
on(computer, desk).
on(notebook, desk).
on(plate, kitchen_table).
on(glass, kitchen_table).
on(cup, book).
on(pen, notebook).

% Transitive "on" relation: if X is on Y, and Y is on Z,
% then X is also (transitively) on Z.
% Example: a cup on a book on a table is also on the table.
on_transitive(X, Y) :- on(X, Y).
on_transitive(X, Y) :-
    on(X, Z),
    on_transitive(Z, Y).

% =====================================================
% 4. NEXT_TO RELATIONS
% =====================================================

next_to(tv, bookshelf).
next_to(sofa, coffee_table).
next_to(coffee_table, armchair).
next_to(armchair, bookshelf).
next_to(bed, wardrobe).
next_to(desk, chair).
next_to(fridge, stove).
next_to(stove, kitchen_table).
next_to(kitchen_table, dishwasher).
next_to(shower, toilet).
next_to(toilet, sink).

% Symmetric version of next_to
neighbor(X, Y) :- next_to(X, Y).
neighbor(X, Y) :- next_to(Y, X).

% =====================================================
% 5. BETWEEN RELATION
% =====================================================
% Y is between X and Z
between(X, Y, Z) :-
    neighbor(X, Y),
    neighbor(Y, Z),
    X \= Z.

% =====================================================
% 6. SAME_ROOM RELATION
% =====================================================
    same_room(O1, O2) :-
        object(O1, Room),
        object(O2, Room),
        O1 \= O2.

% =====================================================
% 7. DYNAMIC OPERATIONS - ADD/REMOVE/MOVE OBJECTS
% =====================================================

% Add a new object
add_object(Obj, Room) :-
    room(Room),
    \+ object(Obj, Room), !,
    assertz(object(Obj, Room)),
    format("[+] '~w' added to '~w'.~n", [Obj, Room]).
add_object(Obj, Room) :-
    format("[!] '~w' already exists in '~w' or room is invalid.~n", [Obj, Room]).

% Remove an object
remove_object(Obj, Room) :-
    object(Obj, Room), !,
    retract(object(Obj, Room)),
    format("[-] '~w' removed from '~w'.~n", [Obj, Room]).
remove_object(Obj, Room) :-
    format("[!] '~w' not found in '~w'.~n", [Obj, Room]).

% Move an object from one room to another
move_object(Obj, FromRoom, ToRoom) :-
    remove_object(Obj, FromRoom),
    add_object(Obj, ToRoom).

% =====================================================
% 8. LOCATION MANAGEMENT (where the agent is)
% =====================================================

% Starting location: corridor
:- assertz(location(me, corridor)).

current_room(Room) :- location(me, Room).

% =====================================================
% 9. NAVIGATION - PATH FINDING (Recursion + Lists)
% =====================================================

% path(Start, Goal, Path)
% Finds a path between two rooms.
path(Start, Goal, Path) :-
    path_helper(Start, Goal, [Start], Path).

% Goal reached - reverse visited list to get correct order
path_helper(Goal, Goal, Visited, Path) :-
    reverse(Visited, Path), !.

% Move to next room, checking it hasn't been visited
path_helper(Current, Goal, Visited, Path) :-
    adjacent_room(Current, Next),
    \+ member(Next, Visited),
    path_helper(Next, Goal, [Next | Visited], Path).

% Find all possible paths and return the shortest
shortest_path(Start, Goal, Shortest) :-
    findall(P, path(Start, Goal, P), AllPaths),
    shortest_in_list(AllPaths, Shortest).

% Helper: pick the shortest list from a list of lists
shortest_in_list([L], L) :- !.
shortest_in_list([L1, L2 | Rest], Shortest) :-
    length(L1, Len1),
    length(L2, Len2),
    (   Len1 =< Len2
    ->  shortest_in_list([L1 | Rest], Shortest)
    ;   shortest_in_list([L2 | Rest], Shortest)
    ).

% =====================================================
% 10. SPECIFIC NAVIGATION TASK - "Go to TV"
% =====================================================

% Go near a specific object (general navigation)
go_near(Obj) :-
    object(Obj, TargetRoom), !,
    location(me, CurrentRoom),
    (   CurrentRoom = TargetRoom
    ->  format(">> Already in the same room as '~w' (~w).~n", [Obj, TargetRoom])
    ;   shortest_path(CurrentRoom, TargetRoom, Path),
        format(">> Going to '~w'.~n", [Obj]),
        format(">> Path: ~w~n", [Path]),
        walk(Path),
        format(">> Arrived! Now near '~w'.~n", [Obj])
    ).
go_near(Obj) :-
    format("[!] '~w' not found on the map.~n", [Obj]).

% Walk along the path - RECURSIVE
walk([_]) :- !.
walk([Room1, Room2 | Rest]) :-
    format("   ~w --> ~w~n", [Room1, Room2]),
    retract(location(me, Room1)),
    assertz(location(me, Room2)),
    walk([Room2 | Rest]).

% Convenience shortcuts
go_to_tv :- go_near(tv).
go_to_bed :- go_near(bed).
go_to_kitchen :- go_near(fridge).

% =====================================================
% 11. LIST OPERATIONS
% =====================================================

% List all objects in a room
objects_in_room(Room, Objects) :-
    findall(O, object(O, Room), Objects).

% Count objects in a room
object_count(Room, Count) :-
    objects_in_room(Room, List),
    length(List, Count).

% Get full apartment inventory (object-room pairs)
full_inventory(Inventory) :-
    findall(O-R, object(O, R), Inventory).

% Check if a room is empty
empty_room(Room) :-
    room(Room),
    \+ object(_, Room).

% Does a room contain a specific object? (list search)
room_contains(Room, Obj) :-
    objects_in_room(Room, List),
    member(Obj, List).

% Find all rooms containing a given object
where_is(Obj, Rooms) :-
    findall(R, object(Obj, R), Rooms).

% =====================================================
% 12. REPORTS AND SUMMARIES
% =====================================================

room_report(Room) :-
    room(Room),
    objects_in_room(Room, Objects),
    length(Objects, Count),
    format("--- ~w ---~n", [Room]),
    format("  Object count : ~w~n", [Count]),
    format("  Objects      : ~w~n~n", [Objects]).

full_report :-
    write('==============================================='), nl,
    write('         APARTMENT STATUS REPORT'), nl,
    write('==============================================='), nl, nl,
    forall(room(R), room_report(R)),
    location(me, MyLocation),
    format("Current location: ~w~n", [MyLocation]),
    write('==============================================='), nl.

% =====================================================
% HELP MESSAGE
% =====================================================

help :-
    nl,
    write('=== AVAILABLE COMMANDS ==='), nl,
    write('?- full_report.                            - Show apartment status'), nl,
    write('?- objects_in_room(kitchen, L).            - List objects in kitchen'), nl,
    write('?- next_to(tv, X).                         - What is next to the TV?'), nl,
    write('?- between(sofa, coffee_table, armchair).  - Is coffee_table between sofa and armchair?'), nl,
    write('?- same_room(tv, sofa).                    - Are they in the same room?'), nl,
    write('?- add_object(table_lamp, living_room).    - Add a new object'), nl,
    write('?- move_object(book, living_room, bedroom).- Move an object'), nl,
    write('?- shortest_path(bathroom, kitchen, P).    - Shortest path between rooms'), nl,
    write('?- go_to_tv.                               - Navigate to the TV!'), nl,
    write('?- go_near(fridge).                        - Navigate to any object'), nl,
    write('?- current_room(R).                        - Where am I now?'), nl,
    nl.

% Auto message on load
:- nl, write('Apartment project loaded. Type "help." for commands.'), nl, nl.
