def generate_square_aux(self, image, res):
        """
        Berechnet die Fourier-Transformation des Bildes und wendet die Frequenzmaske an.
        Speichert das Ergebnis direkt in 'res', um die Struktur des Originals beizubehalten.
        """
        # Berechnung der Fourier-Transformation des Bildes
        fft_image = fft2(image)
        fft_image = fftshift(fft_image)  # Zentrieren der Frequenzen

        # Erstellen der Frequenzmaske (1-Norm)
        grid_size = image.shape[0]  # Quadratisches Bild wird vorausgesetzt
        freqs = np.fft.fftfreq(grid_size, d=1 / grid_size)
        freq_x, freq_y = np.meshgrid(freqs, freqs, indexing="ij")
        freq_norms = np.abs(freq_x) + np.abs(freq_y)

        # Frequenzmaske erstellen und anwenden
        mask = freq_norms <= self.cut_off
        res[:, :] = fft_image * mask  # Ergebnis in res speichern

        # Optional: Normierung der Werte
        if self.normalization:
            res[:, :] /= np.sum(mask)

    def integrate_on_pixel_grid(self, grid_size):
        """
        Erzeugt ein Dummy-Bild und berechnet die maskierte Fourier-Repräsentation im Frequenzraum.
        """
        # Dummy-Bild initialisieren
        dummy_image = np.zeros((grid_size, grid_size))

        # Ergebnis-Array initialisieren
        res = np.zeros_like(dummy_image, dtype=complex)  # Frequenzraum ist komplex

        # Aufruf der Hilfsfunktion
        self.generate_square_aux(dummy_image, res)

        return res
