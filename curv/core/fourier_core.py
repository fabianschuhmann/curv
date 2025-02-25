import numpy as np



class Fourier_Series_Function:
    def __init__(self, Lx, Ly, Nx, Ny):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.q0 = 2 * np.pi / Lx  # Fundamental frequency in x
        self.p0 = 2 * np.pi / Ly  # Fundamental frequency in y
        
        # Fourier coefficients initialized to zeros
        self.Anm = np.zeros((2 * Nx + 1, 2 * Ny + 1))
    
    def getAnm(self):
        if self.Anm is None:
            raise ValueError("Fourier coefficients (Anm) have not been initialized. Call Fit first.")
        return self.Anm

    def Z(self, x, y):
        g = 0
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                g += self.Anm[idx_i, idx_j] * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) +
                    np.sin(self.q0 * i * x + self.p0 * j * y)
                )
        return g

    def Zx(self, x, y):
        gx = np.zeros_like(x)  # Initialize a 2D array for the derivative in x
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                gx += self.Anm[idx_i, idx_j] * self.q0 * i * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) - 
                    np.sin(self.q0 * i * x + self.p0 * j * y)
                )
        return gx

    def Zy(self, x, y):
        gy = np.zeros_like(y)  # Initialize a 2D array for the derivative in y
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                gy += self.Anm[idx_i, idx_j] * self.p0 * j * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) -  # Vectorized
                    np.sin(self.q0 * i * x + self.p0 * j * y)    # Vectorized
                )
        return gy

    def Zxx(self, x, y):
        gxx = np.zeros_like(x)  # Initialize a 2D array for the second derivative in x
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                gxx += self.Anm[idx_i, idx_j] * self.q0 ** 2 * i ** 2 * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) +  # Vectorized
                    np.sin(self.q0 * i * x + self.p0 * j * y)    # Vectorized
                )
        return gxx

    def Zyy(self, x, y):
        gyy = np.zeros_like(y)  # Initialize a 2D array for the second derivative in y
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                gyy += self.Anm[idx_i, idx_j] * self.p0 ** 2 * j ** 2 * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) +  # Vectorized
                    np.sin(self.q0 * i * x + self.p0 * j * y)    # Vectorized
                )
        return gyy

    def Zxy(self, x, y):
        gxy = np.zeros_like(y)  # Initialize a 2D array for the mixed second derivative Zxy
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                gxy += self.Anm[idx_i, idx_j] * self.q0 * self.p0 * i * j * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) +  # Vectorized
                    np.sin(self.q0 * i * x + self.p0 * j * y)    # Vectorized
                )
        return gxy

    def Curv(self, x, y):
        fx = self.Zx(x, y)
        fy = self.Zy(x, y)
        fxx = self.Zxx(x, y)
        fyy = self.Zyy(x, y)
        fxy = self.Zxy(x, y)

        # Mean curvature equation
        numerator = (1 + fx**2) * fyy - 2 * fx * fy * fxy + (1 + fy**2) * fxx
        denominator = (1 + fx**2 + fy**2)**(3/2)
    
        C = numerator / denominator
        return -C

    def Update_coff(self, coff1,coff2):
         self.Anm = 0.5*(coff1+coff2)

    def Fit(self, Data_3M):
        if Data_3M.shape[0] != 3:
            raise ValueError("Data_3M must have shape (3, M).")

        M = Data_3M.shape[1]
        Length = (2 * self.Nx + 1) * (2 * self.Ny + 1)
        
        A = np.zeros((M, Length))
        b = Data_3M[2, :]
        
        index = 0
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                for k in range(M):
                    x, y = Data_3M[0, k], Data_3M[1, k]
                    A[k, index] = (
                        np.cos(self.q0 * i * x + self.p0 * j * y) +
                        np.sin(self.q0 * i * x + self.p0 * j * y)
                    )
                index += 1
        
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        self.Anm = coeffs.reshape((2 * self.Nx + 1, 2 * self.Ny + 1))



if __name__=="__main__":
    pass
