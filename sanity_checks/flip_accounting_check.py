import numpy as np
from randPageSelector import Selector



class FlipAccount:
    def __init__(self, pathBitmap: str, pathPrev: str, pathCurr: str, numPages: int, calculatedHammingPath: str):
        self.selector = Selector(pathBitmap, numPages)
        self.pathCurr = pathCurr
        self.pathPrev = pathPrev
        self.numPages = numPages
        self.hammingPath = calculatedHammingPath

        self.selectedAll = None
        self.selectedActive = None
        self.selectedInactive = None

        self.LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8) #Lookup Table

    def shuffle(self, mode="active"):
        mode = mode.lower()
        if mode == "all":
            self.selectedAll = self.selector.select_random_all(self.numPages)
        elif mode == "active":
            self.selectedActive = self.selector.select_random_active(self.numPages)
        elif mode == "inactive":
            self.selectedInactive = self.selector.select_random_inactive(self.numPages)
        else:
            raise ValueError("mode must be 'all', 'active', or 'inactive'")

    # def shuffle_all_choice(self):
    #     self.selectedAll = self.selector.select_random_all(self.numPages)

    # def shuffle_active_choice(self):
    #     self.selectedActive = self.selector.select_random_active(self.numPages)

    # def shuffle_inactive_choice(self):
    #     self.selectedInactive = self.selector.select_random_inactive(self.numPages)

    @staticmethod
    def load_hamming_file(path: str) -> np.ndarray:
        """
        Load the Hamming text file as a NumPy array of integers.
        Each line corresponds to the bit-flip count for one active page.
        """
        # # Handles text file with one value per line
        # with open(path, "r") as f:
        #     data = [int(line.strip()) for line in f if line.strip()]
        # return np.array(data, dtype=np.uint32)
    
        return np.loadtxt(path, dtype=np.uint32)
    
    

    def compare(self, mode="active"):
        """
        Compare pages between prev and curr snapshot for the given mode.
        Mode can be: 'all', 'active', 'inactive'.
        """
        mode = mode.lower()
        hamming_values = self.load_hamming_file(self.hammingPath)

        if mode == "all":
            index_choice = self.selectedAll
        elif mode == "active":
            index_choice = self.selectedActive
        elif mode == "inactive":
            index_choice = self.selectedInactive
        else:
            raise ValueError("mode must be 'all', 'active', or 'inactive'")

        if index_choice == None or len(index_choice) == 0:
            raise ValueError(f"No indices selected for mode '{mode}'. Run shuffle(mode='{mode}') first.")
        
        subset_hamming = hamming_values[index_choice]
        
        prev_pages = self.selector.read_pages(path=self.pathPrev, indices=index_choice)
        curr_pages = self.selector.read_pages(path=self.pathCurr, indices=index_choice)

        xor_vector = np.bitwise_xor(prev_pages, curr_pages)
        recomputed_hamming = self.LUT[xor_vector].sum(axis=1)

        diff = recomputed_hamming.astype(np.int64) - subset_hamming.astype(np.int64)
        abs_err = np.abs(diff)

        mae = abs_err.mean()
        mse = (diff**2).mean()
        rmse = np.sqrt(mse)
        mape = np.mean(abs_err / np.maximum(subset_hamming, 1)) * 100  # avoid divide-by-zero
        rel_max_err = abs_err.max() / np.maximum(subset_hamming.mean(), 1)


        print(
        f"[{mode.upper()}] MAE={mae:.4f}, RMSE={rmse:.4f}, "
        f"MAPE={mape:.2f}%, MaxErr={abs_err.max()} ({rel_max_err:.3%} rel)"
        )

        return {
            "mode": mode,
            "diff": diff,
            "abs_err": abs_err,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "max_err": int(abs_err.max()),
            "rel_max_err": rel_max_err,
        }


        

    # def compare_all(self):
    #     prev_pages = self.selector.read_pages(path=self.pathPrev, indices=self.selectedAll)
    #     curr_pages = self.selector.read_pages(path=self.pathCurr, indices=self.selectedAll)
    #     np.bitwise_xor(prev_pages, curr_pages)
    #     pass
    
    # def compare_active(self):
    #     prev_pages = self.selector.read_pages(path=self.pathPrev, indices=self.selectedActive)
    #     curr_pages = self.selector.read_pages(path=self.pathCurr, indices=self.selectedActive)
    #     np.bitwise_xor(prev_pages, curr_pages)
    #     pass

    # def compare_inactive(self):
    #     prev_pages = self.selector.read_pages(path=self.pathPrev, indices=self.selectedInactive)
    #     curr_pages = self.selector.read_pages(path=self.pathCurr, indices=self.selectedInactive)
    #     np.bitwise_xor(prev_pages, curr_pages)
    #     pass




        