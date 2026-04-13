import cv2
import numpy as np
from pydmd import DMD
import matplotlib.pyplot as plt

def vid_proc(path):
    """
    Loads a video file, converts each frame to grayscale, flattens it, 
    and constructs a data matrix X suitable for DMD.
    
    Parameters:
        video_path (str): Path to the video file (e.g., 'traffic.mp4')
    
    Returns:
        X (ndarray): 2D data matrix of shape (pixels, frames)
        frame_shape (tuple): Original (height, width) of grayscale frames
    """
    # Open the video file
    cap = cv2.VideoCapture(path)
    
    # List to store flattened grayscale frames
    frames = []
    
    # Will store the (height, width) of frames
    frame_shape = None

    while True:
        ret, frame = cap.read()  # Read next frame
        if not ret:
            break  # Stop if no frame is returned (end of video)

        # Convert to grayscale (1 channel)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Store shape of first frame
        if frame_shape is None:
            frame_shape = gray.shape

        # Flatten the 2D frame into 1D vector and store it
        frames.append(gray.flatten())

    # Release the video file
    cap.release()

    # Create data matrix: shape = (pixels, frames)
    X = np.array(frames).T

    return X, frame_shape

def dmd(X, rank = None):
    """
    Compute the Dynamic Mode Decomposition (DMD) of a data matrix X.

    Parameters:
        X (ndarray): Data matrix of shape (n_features, n_snapshots),
                     where each column is a snapshot in time.
        rank (int or None): Optional truncation rank for dimensionality reduction.
                            If None, full rank is used.

    Returns:
        Phi (ndarray): DMD modes (n_features x r)
        omega (ndarray): Continuous-time eigenvalues (r,)
        b (ndarray): Mode amplitudes (r,)
        X_dmd (ndarray): Low-rank reconstruction of X using DMD
    """
    # Step 1: Build time-shifted data matrices
    X1 = X[:, :-1]  # All columns except last
    X2 = X[:, 1:]   # All columns except first

    # Step 2: SVD of X1 → X1 ≈ U Σ V*
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)

    if rank is not None:
        # Truncate to rank-r
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

    # Step 3: Build low-rank A tilde
    S_inv = np.diag(1 / S)
    A_tilde = U.T @ X2 @ Vh.T @ S_inv  # Ã = Uᵗ X2 V Σ⁻¹

    # Step 4: Eigendecomposition of A_tilde
    eigvals, W = np.linalg.eig(A_tilde)

    # Step 5: Compute DMD modes Φ
    Phi = X2 @ Vh.T @ S_inv @ W  # DMD modes

    # Step 6: Compute time dynamics
    # Continuous-time eigenvalues
    dt = 1  # assume unit timestep; change if needed
    omega = np.log(eigvals) / dt

    # Initial condition projection
    x1 = X[:, 0]
    b = np.linalg.lstsq(Phi, x1, rcond=None)[0]

    # Step 7: Reconstruct the data using DMD
    time_points = X.shape[1]
    time_dynamics = np.zeros((len(omega), time_points), dtype=complex)
    for i in range(time_points):
        time_dynamics[:, i] = b * np.exp(omega * i * dt)

    X_dmd = Phi @ time_dynamics

    return Phi, omega, b, X_dmd.real

def background(X, X_dmd, threshold=20):
    """
    Compute sparse foreground and low-rank background from DMD reconstruction.

    Parameters:
        X (ndarray): Original data matrix
        X_dmd (ndarray): DMD reconstruction (low-rank background)
        threshold (float): Threshold for noise removal in foreground

    Returns:
        X_background (ndarray): Low-rank background
        X_foreground (ndarray): Sparse foreground (thresholded)
    """
    X_sparse = X - X_dmd
    X_foreground = np.abs(X_sparse)
    X_foreground[X_foreground < threshold] = 0
    return X_dmd, X_foreground

def visualize_frame(X, X_background, X_foreground, frame_shape, frame_index):
    """
    Visualize original, background, and foreground for a selected frame.

    Parameters:
        X (ndarray): Original data matrix
        X_background (ndarray): Low-rank background
        X_foreground (ndarray): Sparse foreground
        frame_shape (tuple): (height, width) of frames
        frame_index (int): Index of the frame to visualize
    """
    original = X[:, frame_index].reshape(frame_shape)
    background = X_background[:, frame_index].reshape(frame_shape)
    foreground = X_foreground[:, frame_index].reshape(frame_shape)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Frame")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Low-Rank Background")
    plt.imshow(background, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Sparse Foreground")
    plt.imshow(foreground, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def save_video(X_data, frame_shape, filename, fps=24, normalize=True):
    """
    Save a sequence of frames from a data matrix as a grayscale video.

    Parameters:
        X_data (ndarray): Data matrix of shape (pixels, time)
        frame_shape (tuple): (height, width) of each frame
        filename (str): Output filename (e.g., "foreground.avi")
        fps (int): Frames per second for the output video
        normalize (bool): Whether to normalize frames to 0–255
    """
    height, width = frame_shape
    n_frames = X_data.shape[1]

    # Define video writer (MJPG is widely supported)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=False)

    for i in range(n_frames):
        frame = X_data[:, i].reshape(frame_shape)
        if normalize:
            frame = frame - frame.min()
            frame = 255 * frame / frame.max() if frame.max() != 0 else frame
        frame_uint8 = frame.astype(np.uint8)
        out.write(frame_uint8)

    out.release()
    print(f"✅ Video saved as '{filename}'")