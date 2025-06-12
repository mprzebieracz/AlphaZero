# cython: language_level=3
# distutils: language = c++
from libcpp.map cimport map as cpp_map
from libcpp.utility cimport pair
from libc.math cimport sqrt
cimport numpy as np

cdef cpp_map[int, Node*] temp_map

cdef class Node:
    def __cinit__(self, object game_state, Node parent=None, float prior=0):
        self.game_state = game_state
        self._parent = <Node*>parent if parent is not None else NULL
        self._children = new cpp_map[int, Node*]()
        self._N = 0
        # self._children.swap(temp_map)
        self.W = 0.0
        self.P = prior

    cdef float get_Q(self) nogil:
        return self.W / self._N if self._N > 0 else 0.0

    cdef float get_U(self, float exploration_weight) nogil:
        if self._parent != NULL:
            return self.P * sqrt(self._parent._N) / (1 + self._N) * exploration_weight
        return 0.0

    cdef float get_UCB(self, float exploration_weight) nogil:
        return self.get_Q() + self.get_U(exploration_weight)

    cpdef tuple select_child(self, float exploration_weight=1.0):
        """
        Returns (action, Node)
        """
        cdef pair[int, Node*] p
        cdef Node* best_node = NULL
        cdef float best_score = -1e9
        cdef float score
        cdef int best_action = -1

        with nogil:
            for p in self._children:
                score = p.second.get_UCB(exploration_weight)
                if score > best_score:
                    best_score = score
                    best_node = p.second
                    best_action = p.first

        if best_node == NULL:
            return None
        return (best_action, <object>best_node)

    cpdef void expand(self, np.ndarray policy):
        cdef int i, size = policy.shape[0]
        cdef float p
        cdef Node* child_node
        cdef object new_game_state
        cdef float[:] policy_view = policy

        for i in range(size):
            p = policy_view[i]
            if p == 0:
                continue

            new_game_state = self.game_state.clone()
            new_game_state.step(i)
            child_node = Node(new_game_state, parent=self, prior=p)
            self._children[i] = child_node

    # Optional helper: Not exposed, just internal
    cdef int num_visits(self) nogil:
        return self._N

    cdef void increment_visits(self, int val) nogil:
        self._N += val

    cpdef bint is_expanded(self):
        return self._children.size() > 0

    cpdef bint is_terminal(self):
        return self.game_state.is_terminal()
