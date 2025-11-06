import os
import numpy as np
from randPageSelector import Selector

class SymmetryNullValidator:
    def __init__(self, pathBitmap: str, pathPrev: str, pathCurr: str, numPages: int, calculatedHammingPath: str, unitMEMsize: int = 4096):
        self.selector = Selector(pathBitmap, numPages)
        self.pathCurr = pathCurr
        self.pathPrev = pathPrev
        self.numPages = numPages
        self.hammingPath = calculatedHammingPath
        self.MEMUNITsize = unitMEMsize

        self.LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8) #Lookup Table
    
    def mm_open(self, path: str) -> np.memmap:
        expected_size = self.numPages
        actual_size = os.path.getsize(path)
        if actual_size != expected_size:
            raise ValueError(f"File size mismatch: expected {expected_size} bytes, got {actual_size} bytes.")
        
        n_pages = actual_size // self.MEMUNITsize
        if actual_size % self.MEMUNITsize != 0:
            raise ValueError(f"File size {actual_size} is not a multiple of page size {self.MEMUNITsize}.")
        
        mm = np.memmap(path, dtype=np.uint8, mode='r', shape=(n_pages,self.MEMUNITsize))
        return mm, n_pages

    def calc_hamming(self, prev, curr):
        xor = np.bitwise_xor(prev, curr)
        return self.LUT[xor].sum(axis=1, dtype=np.uint32)
    
    #if two pages are both null, we consider their hamming distance to be zero
    def test_nullity(self, path: str | None = None, indices: np.array | None=None):
        if path is None:
            path = self.pathCurr
        mm, n_pages = self.mm_open(path)
        if indices is None:
            indices = np.arange(n_pages)
        
        pages = np.array(mm[indices], copy=True)
        hamming = self.calc_hamming(pages, pages)

        is_ok = np.all(hamming == 0)

        return {"is_ok": is_ok, "max": int(hamming.max()), "num_pages": len(indices)}
    
    # test if hamming(page1, page2) == hamming(page2, pag1)
    def test_symmetry(self, indices: np.array | None=None, path_curr: str | None = None, path_prev: str | None =None):
        if path_curr is None:
            path_curr = self.pathCurr

        if path_prev is None:
            path_prev = self.pathPrev

        mm_prev, n_pages_prev = self.mm_open(path_prev)
        mm_curr, n_pages_curr = self.mm_open(path_curr)

        if n_pages_prev != n_pages_curr:
            raise ValueError(f"Number of pages mismatch between prev ({n_pages_prev}) and curr ({n_pages_curr}).")

        if indices is None:
            indices = np.arange(n_pages_prev)
        
        pages_prev = np.array(mm_prev[indices], copy=True)
        pages_curr = np.array(mm_curr[indices], copy=True)

        hamming_prev_to_curr = self.calc_hamming(pages_prev, pages_curr)
        hamming_curr_to_prev = self.calc_hamming(pages_curr, pages_prev)

        is_ok = np.all(hamming_prev_to_curr == hamming_curr_to_prev)

        return {"is_ok": is_ok, "max_diff": int(np.max(np.abs(hamming_prev_to_curr - hamming_curr_to_prev))),
                 "num_pages": len(indices), "mae": float(np.mean(np.abs(hamming_prev_to_curr - hamming_curr_to_prev)))}

    # check frozen VM, take to path for snapshots like before and a path for indices like before and check if the change is 0
    def frozen_vm_check(self, indices: np.array | None=None, path_curr: str | None = None, path_prev: str | None =None):
        if path_curr is None:
            path_curr = self.pathCurr

        if path_prev is None:
            path_prev = self.pathPrev

        if indices is None:
            indices = np.arange(self.numPages)

        mm_prev, n_pages_prev = self.mm_open(path_prev)
        mm_curr, n_pages_curr = self.mm_open(path_curr)

        if n_pages_prev != n_pages_curr:
            raise ValueError(f"Number of pages mismatch between prev ({n_pages_prev}) and curr ({n_pages_curr}).")

        pages_prev = np.array(mm_prev[indices], copy=True)
        pages_curr = np.array(mm_curr[indices], copy=True)

        hamming = self.calc_hamming(pages_prev, pages_curr)

        is_ok = np.all(hamming == 0)

        return {
            "is_ok": is_ok,
            "mae": float(hamming.mean()),
            "max": int(hamming.max()),
            "zeros_ratio": float((hamming == 0).mean()),
            "num_pages": len(indices),
            "passed": bool(hamming.max() == 0)
        }
    
    # Generate a page, and run it through T periods of bit flips with given bit_period, and check if the hamming distance is as expected
    def synthetic_periodic_check(self, bit_period: int, T: int = 128):
        """
        Minimal synthetic: 1 page over T steps, toggling a byte with a fixed period.
        Confirms spectral peak at planted frequency.
        """
        # Build a time series of per-interval Hamming counts
        page = np.zeros((1, self.page_size), dtype=np.uint8)
        series = []
        for t in range(T):
            prev = page.copy()
            # flip one byte every 'bit_period' steps (just for demonstration)
            if t % bit_period == 0:
                page[0, 0] ^= 0xFF
            curr = page
            H = self._hamming_pages(prev, curr)[0]
            series.append(H)

        x = np.array(series, dtype=np.float32)
        # FFT magnitude
        X = np.abs(np.fft.rfft(x - x.mean()))
        peak_bin = int(np.argmax(X[1:]) + 1)  # skip DC
        return {"peak_bin": peak_bin, "series": x.tolist(), "fft_len": int(len(X))}

