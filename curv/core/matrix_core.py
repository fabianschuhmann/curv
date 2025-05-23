import numpy as np



class Matrix_Function:
    def __init__(self, Lx, Ly, Nx, Ny):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.q0 = 2 * np.pi / Lx  # Fundamental frequency in x
        self.p0 = 2 * np.pi / Ly  # Fundamental frequency in y
        self.total_modes = (2 * Nx + 1) * (2 * Ny + 1)
        
        # Initialized Matrices
        self.Hm_matrix = np.zeros((self.total_modes, self.total_modes))
        self.Hp_matrix = np.zeros((self.total_modes, self.total_modes))
        self.Kp_matrix = np.zeros((self.total_modes, self.total_modes))
        self.C_vector= np.zeros((self.total_modes, 1))
        self.sigmaA_matrix = np.zeros((self.total_modes, self.total_modes))
        self.A_vector = np.zeros((self.total_modes, 1))
        self.A_matrix= np.zeros((self.total_modes, self.total_modes))
        self.q_vector = np.zeros((self.total_modes, 2))
        
    

    def get_A_vector(self, A_vector):
        self.A_vector = A_vector

    def get_A_matrix(self, A_matrix):
        self.A_matrix = A_matrix

    def make_sigmaA_matrix(self):
        for i in range(self.total_modes):
            for j in range(self.total_modes):
                self.sigmaA_matrix[i, j] = (
                    self.A_matrix[i, j] - self.A_vector[i] * self.A_vector[j]
                )

    def make_q_vector(self):
        index = 0
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                
                self.q_vector[index, 0] = self.q0 * i
                self.q_vector[index, 1] = self.p0 * j
                index += 1
    
    def make_Hm_matrix(self):

        for i in range(self.total_modes):
            for j in range(self.total_modes):
                if i==j:
                    qi2=self.q_vector[i,0]**2+self.q_vector[i,1]**2
                    qj2=self.q_vector[j,0]**2+self.q_vector[j,1]**2

                    self.Hm_matrix[i,j]=self.Lx*self.Ly*qi2*qj2 




if __name__=="__main__":
    pass
