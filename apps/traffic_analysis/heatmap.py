import cv2
import numpy as np

class HeatmapGenerator:
    """
    Accumulates vehicle centroid trajectories into a continuous movement density grid
    and overlays a colorized heatmap onto the video frame.
    """
    def __init__(self, config, frame_shape: tuple = (720, 1280)):
        self.config = config
        self.height, self.width = frame_shape[:2]
        self.decay = config.heatmap_decay
        self.alpha = config.heatmap_alpha
        
        # 2D Float accumulator grid
        self.density_grid = np.zeros((self.height, self.width), dtype=np.float32)

    def update(self, centroids: list) -> None:
        """
        Accumulate current frame centroids onto density grid with temporal decay.
        """
        # Apply exponential decay to past positions
        self.density_grid *= self.decay

        # Add Gaussian spot for each active vehicle center point
        for (cx, cy) in centroids:
            ix, iy = int(cx), int(cy)
            if 0 <= ix < self.width and 0 <= iy < self.height:
                # Add gaussian accumulation around centroid
                y_min, y_max = max(0, iy - 15), min(self.height, iy + 15)
                x_min, x_max = max(0, ix - 15), min(self.width, ix + 15)
                self.density_grid[y_min:y_max, x_min:x_max] += 1.0

    def generate_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Produce colorized heatmap and blend over the original frame.
        """
        if not self.config.enable_heatmap:
            return frame

        # Normalize density grid to [0, 255]
        max_val = np.max(self.density_grid)
        if max_val <= 0:
            return frame

        norm_grid = (self.density_grid / max_val * 255.0).astype(np.uint8)
        
        # Apply Gaussian Blur to smooth heatmap density
        blurred = cv2.GaussianBlur(norm_grid, (21, 21), 0)
        
        # Apply JET colormap
        color_heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

        # Create mask so zero-density background remains unaltered
        mask = blurred > 10

        out_frame = frame.copy()
        out_frame[mask] = cv2.addWeighted(frame[mask], 1.0 - self.alpha, color_heatmap[mask], self.alpha, 0)
        return out_frame
