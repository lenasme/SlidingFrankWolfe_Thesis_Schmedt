def compute_obs(self, cut_f, grid_size, version=0):
        from Setup.ground_truth import EtaObservation

        if self.num_atoms == 0:
            print("atoms Liste ist leer")
            return np.zeros((grid_size , grid_size))

        if version == 0:
            combined_image = np.zeros((grid_size, grid_size))
            fourier = EtaObservation(cut_f)
            for atom in self.atoms:
                atom_simple_function = SimpleFunction(atom, imgsz= grid_size)
                atom_image = atom_simple_function.transform_into_image(grid_size)
                print(atom_image)
                combined_image += atom.weight * atom_image
            truncated_transform = fourier.trunc_fourier(combined_image)
            return np.real(truncated_transform)

        elif version == 1:
            observations = []
            fourier = EtaObservation(cut_f)
            for atom in self.atoms:
                atom_simple_function = SimpleFunction(atom, imgsz = grid_size)
                atom_image = atom_simple_function.transform_into_image(grid_size)
                truncated_transform = fourier.trunc_fourier(atom.weight * atom_image)
                observations.append(np.real(truncated_transform))
            return np.array(observations)
        else:
            raise ValueError("Invalid version specified. Use version=0 or version=1.")


 #def compute_phi_E(self, cut_f):
     #   num_freqs = 2 * cut_f + 1  # Anzahl der Frequenzen in jede Richtung
      #  phi_e_matrix = np.zeros((len(self.atoms), num_freqs ** 2))  # Matrix für Ergebnisse
       # dx = dy = 1 / (self.imgsz - 1)  # Diskretisierungsschritte

        #for atom_index, atom in enumerate(self.atoms):
            # Transformiere das Atom in ein Bild
         #   simple_func = SimpleFunction([atom], imgsz=self.imgsz)
          #  test_func_im = simple_func.transform_into_image(self.imgsz)
            
           # index = 0
            #for k1 in range(-cut_f , cut_f+1):
             #   for k2 in range(-cut_f , cut_f+1):
              #      # Erstelle Gitter für (x, y)
               #     x = np.linspace(0, 1, self.imgsz)
                #    y = np.linspace(0, 1, self.imgsz)
                 #   X, Y = np.meshgrid(x, y, indexing='ij')

                    # Berechne cosinusbasierte Gewichtung
                  #  cos_part = np.cos(2 * np.pi * (k1 * X + k2 * Y))

                    # Berechne das gewichtete Integral
                   # weighted_integral = np.sum(test_func_im * cos_part) * dx * dy

                    # Speichere das Ergebnis in der Matrix
                    #phi_e_matrix[atom_index, index] = weighted_integral
                    #index += 1

        #return phi_e_matrix