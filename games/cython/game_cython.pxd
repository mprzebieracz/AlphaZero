# cython: language_level=3
# game_cython.pxd
cimport torch

cdef class Hashable:
    # Dummy placeholder since you didn't provide details
    pass

cdef class Game:
    cdef int action_dim
    cdef tuple state_dim

    cpdef reset(self)
    cpdef object get_action_size(self)
    cpdef list get_legal_actions(self)
    cpdef step(self, object action)
    cpdef bint is_terminal(self)
    cpdef object reward(self)
    cpdef object get_canonical_state(self)
    cpdef Game clone(self)
    cpdef render(self)

cdef class GameState(Hashable):
    cpdef object get_canonical_state(self)
