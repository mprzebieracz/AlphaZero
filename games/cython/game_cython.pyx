# cython: language_level=3

cdef class Hashable:
    pass


cdef class Game:
    # cdef int action_dim
    # cdef tuple state_dim

    cpdef reset(self):
        raise NotImplementedError()

    cpdef object get_action_size(self):
        raise NotImplementedError()

    cpdef list get_legal_actions(self):
        raise NotImplementedError()

    cpdef step(self, object action):
        raise NotImplementedError()

    cpdef bint is_terminal(self):
        raise NotImplementedError()

    cpdef object reward(self):
        raise NotImplementedError()

    cpdef object get_canonical_state(self):
        raise NotImplementedError()

    cpdef Game clone(self):
        raise NotImplementedError()

    cpdef render(self):
        raise NotImplementedError()


cdef class GameState(Hashable):
    cpdef object get_canonical_state(self):
        raise NotImplementedError()
