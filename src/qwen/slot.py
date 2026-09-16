

# request slot management for sampling parameters and sampling state
class RequestSlotPool:
    def __init__(self, capacity: int):  # capacity == max_num_seqs
        self.capacity = capacity
        self._free = list(range(capacity - 1, -1, -1))   # [capacity-1, ..., 2, 1, 0], pop() -> lowest first

    def alloc(self) -> int:
        return self._free.pop()          # IndexError if exhausted: capacity == max_num_seqs

    def free(self, req) -> None:
        if req.slot is not None:
            self._free.append(req.slot)
            req.slot = None
