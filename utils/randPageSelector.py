import numpy as np
import os
from bitmap_utils import load_bitmap, active_page_indices, inactive_page_indices

class Selector:
    def __init__(self, path: str, num_pages: int, seed: int = 1337, MEMUNIT_size: int = 4096):
        self.bitmap = load_bitmap(path, num_pages)
        self.actives = active_page_indices(self.bitmap)
        self.inactives = inactive_page_indices(self.bitmap)
        self.MEMUNIT_size = MEMUNIT_size

        self.numPages = num_pages

        self.perm = None
        self.invPerm = None
        self.permMap = None

        # Create an independent Generator object
        self.rng = np.random.default_rng(seed)

        self.chosenAll = None
        self.selectionAll = None

        self.chosenActive = None
        self.selectionActive = None

        self.chosenInactive = None
        self.selectionInactive = None

    def select_random_all(self,amount)->np.ndarray:
        self.chosen_all = self.rng.choice(np.arange(self.numPages), size=min(amount, len(self.pages)), replace=False)
        return self.chosen_all

    def select_random_active(self, amount)->np.ndarray:
        if len(self.actives) == 0:
            return np.array([], dtype=int)
        self.chosen_all = self.rng.choice(self.actives, size=min(amount, len(self.actives)), replace=False)

    def select_random_inactive(self, amount)->np.ndarray:
        if len(self.inactives) == 0:
            return np.array([], dtype=int)
        self.chosen_all = self.rng.choice(self.inactives, size=min(amount, len(self.inactives)), replace=False)

    def get_chosenAll(self)->np.ndarray:
        if not self.chosenAll:
            return np.array([], dtype=int)
        return self.chosenAll
    
    def get_chosenActive(self)->np.ndarray:
        if not self.chosenActive:
            return np.arra([], dtype=int)
        return self.chosenActive
    
    def get_chosenInactive(self)->np.ndarray:
        if not self.chosenInactive:
            return np.array([], dtype=int)
        return self.chosenInactive
    
    def read_pages(self, path: str, indices: np.array)->np.ndarray:
        """Return pages[ len(indices) x page_size ] as uint8 (copy).
        Only touches the requested pages on disk."""
        file_size = os.path.getsize(path)
        assert file_size % self.MEMUNIT_size == 0
        num_MUs = file_size // self.MEMUNIT_size

        mm = np.memmap(path, mode = 'r', dtype= np.uint32, shape=(num_MUs, self.MEMUNIT_size))
        requested_pages = mm[indices]
        return np.array(requested_pages, copy=True)
    
    def generate_block_permutation(self, seed: int | None = None):
        """
        Generate a random permutation for a block of given size.

        Args:
            block_pages : int
                Number of pages in the block (e.g., 256)
            seed : int | None
                Random seed for reproducibility

        Returns:
            perm : ndarray[int]
                The permutation of [0..block_pages-1]
            mapping : ndarray[int, 2]
                Mapping table where mapping[i] = [i, perm[i]]
            inv : ndarray[int]
                Inverse permutation, so inv[perm[i]] = i
        """
        rng = np.random.default_rng(seed=seed)
        self.perm = rng.permutation(self.numPages).astype(np.uint64)
        self.invPerm = np.empty_like(self.perm)
        self.invPerm[self.perm] = np.arange(self.numPages, dtype=np.uint64)
        self.permMap = np.stack([np.arange(self.numPages), self.perm], axis=1)
        return self.perm, self.invPerm, self.permMap
    
    def get_perm_vars(self):
        p = self.perm if self.perm else []
        i = self.invPerm if self.invPerm else []
        m = self.permMap if self.permMap else []
        return p, i, m
    
    def read_permuted_pages(self, prevPath: str, currPath: str, startingBlock: int, perm: np.ndarray | None = None):
        """
        Reads the block [block_start_page : block_start_page + block_pages)
        from both snapshots, but in the order defined by 'perm'.

        Returns:
            prev_block_perm, curr_block_perm  : (block_pages, page_size) uint8
            global_indices                    : (block_pages,) global page indices used
        """
        prev_size = os.path.getsize(prevPath)
        total_pages = prev_size // self.MEMUNIT_size

        assert prev_size % self.MEMUNIT_size == 0, "File not aligned to page size"
        assert startingBlock + self.numPages <= total_pages, "Chosen Block Overflows"

        assert self.perm != None, "Permutation is not created, call \"generate_block_permutation\" "
        used_permutation = perm if perm else self.perm

        local_perm_indices = startingBlock + used_permutation

        mm_prev = np.memmap(prevPath, mode='r', dtype=np.uint8, shape=(total_pages, self.MEMUNIT_size))
        mm_curr = np.memmap(currPath, mode='r', dtype=np.uint8, shape=(total_pages, self.MEMUNIT_size))

        prev_perm = np.array(mm_prev[local_perm_indices], copy=True)
        curr_perm = np.array(mm_curr[local_perm_indices], copy=True)

        return prev_perm, curr_perm, local_perm_indices





    


    


    

