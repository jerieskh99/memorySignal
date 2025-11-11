import numpy as np
import os
# from randPageSelector import Selector
from utils.randPageSelector import Selector

class OrderInvariance:
    def __init__(self, pathBitmap: str, pathPrev: str, pathCurr: str, numPages: int, calculatedHammingPath: str):
        self.selector = Selector(pathBitmap, numPages)
        self.pathCurr = pathCurr
        self.pathPrev = pathPrev
        self.numPages = numPages
        self.hammingPath = calculatedHammingPath

        self.LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8) #Lookup Table

    def calc_hamming(self, prev, curr):
        xor = np.bitwise_xor(prev, curr)
        return self.LUT[xor].sum(axis=1, dtype=np.uint32)

    def calcInvarianceMetrics(self, seed: int=42, start_page: int=0):
        # 1. Generate permutation
        perm, inv, mapping = self.selector.generate_block_permutation(seed=seed)

        # 2. Read natural order
        natural_perm = np.arange(self.numPages)
        prev_nat, curr_nat, _ = self.selector.read_permuted_pages(prevPath=self.pathPrev, currPath=self.pathCurr, startingBlock=start_page, perm=natural_perm)

        hamming_natural = self.calc_hamming(prev_nat, curr_nat)

        # 3. Read permuted order
        self.selector.generate_block_permutation(seed=42)
        prev_perm, curr_perm, _ = self.selector.read_permuted_pages(prevPath=self.pathPrev, currPath=self.pathCurr, startingBlock=start_page, perm=perm)

        hamming_perm = self.calc_hamming(prev=prev_perm, curr=curr_perm)
        h_reverted = hamming_perm[inv] # Reorder

        # 4. Compare
        diff = h_reverted.astype(np.uint64) - hamming_natural.astype(np.uint64)
        mae = np.abs(diff).mean()
        ok = mae == 0

        print(f"[C2 proof] block={self.numPages} start={start_page} seed={seed} -> MAE={mae:.6f} ({'PASS' if ok else 'FAIL'})")
        return {"mae": mae, "pass": ok, "mapping": mapping[:10].tolist()}




