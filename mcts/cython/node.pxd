# node.pxd
from libcpp.map cimport map as cpp_map
from games.cython.game_cython cimport Game
cimport numpy as np
# cimport Game 

# node.pxd
cdef class Node:
    cdef Game game_state
    cdef Node* _parent
    cdef cpp_map[int, Node*] _children
    cdef int _N
    cdef float W
    cdef float P

    cpdef void expand(self, np.ndarray policy)
    cpdef tuple select_child(self, float exploration_weight)
    cpdef bint is_expanded(self)
    cpdef bint is_terminal(self)

