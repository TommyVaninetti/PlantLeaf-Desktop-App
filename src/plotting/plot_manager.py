# Copyright (C) 2026 Tommaso Vaninetti
#
# This file is part of PlantLeaf.
#
# PlantLeaf is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# PlantLeaf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with PlantLeaf. If not, see <https://www.gnu.org/licenses/>.

#questa classe serve come base per i grafici e contiene metodi comuni, che sono:
#STILE
#1. Impostazione del titolo del grafico
#2. Impostazione del titolo dell'asse X
#3. Impostazione del titolo dell'asse Y
#4. Impostazione del limite dell'asse X
#5. Impostazione del limite dell'asse Y
#6. Impostazione della griglia
#7. Impostazione del tema (linea, background)

#DATI
#8. Aggiunta di una curva al grafico
#9. Pulizia del grafico
#10. (magari si possono fare anche i calcoli integrati) (da delegare a altro modulo, magari quello del rewatch!!!)

#CREAZIONE DEL WIDGET PLOT
#11. Creazione del widget plot
#(sostituzione integrata in base_window.py)

#IMPOSTAZIONI GENERALI DEL GRAFICO
#13. Disattiva menu contestuale
#14. Aggiungi la griglia



from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np
from PySide6.QtGui import QFont

class TimeAxisItem(pg.AxisItem):
    """
    A bottom axis that prints recording position as H:MM:SS.ss instead of seconds.

    pyqtgraph applies SI prefixes to any axis given a unit, so an axis in seconds
    labels a 110-minute recording in **kiloseconds** ("2 ks", "4 ks"). That is
    correct SI and useless for finding a click: it cannot be compared against the
    playback clock, the click table, or a screenshot filename, all of which are
    H:MM:SS.ss.

    Resolution is chosen from the tick SPACING, not fixed, because the same axis
    class serves both the overview (tick every few minutes — seconds are noise)
    and the iFFT tab (tick every few milliseconds — a frame is 2.56 ms and two
    decimals would collapse adjacent ticks to the same string).
    """

    def tickStrings(self, values, scale, spacing):
        if spacing >= 60.0:
            dec = 0
        elif spacing >= 1.0:
            dec = 0
        elif spacing >= 0.01:
            dec = 2
        else:
            dec = 3

        out = []
        for v in values:
            try:
                total = float(v)
            except (TypeError, ValueError):
                out.append(""); continue
            sign = "-" if total < 0 else ""
            total = abs(total)
            hours, rem = divmod(total, 3600.0)
            minutes, secs = divmod(rem, 60.0)
            # Guard the 59.995 -> "60.00" carry that would print 0:01:60.00
            if round(secs, dec) >= 60.0:
                secs = 0.0
                minutes += 1
                if minutes >= 60:
                    minutes = 0
                    hours += 1
            width = 2 if dec == 0 else dec + 3
            out.append(f"{sign}{int(hours)}:{int(minutes):02d}:{secs:0{width}.{dec}f}")
        return out


class BasePlotWidget(QWidget):
    def __init__(self, x_label: str, y_label: str,
                 x_range: tuple, y_range: tuple,
                 x_min=None, x_max=None, y_min=None, y_max=None,
                 unit_x=None, unit_y=None,
                 parent=None, x_axis_item=None):
        super().__init__(parent)

        # A custom bottom axis has to be handed to the PlotWidget at construction;
        # pyqtgraph does not support swapping it afterwards.
        self.plot_widget = (pg.PlotWidget(axisItems={'bottom': x_axis_item})
                            if x_axis_item is not None else pg.PlotWidget())

        # Impostazioni di stile
        self.set_x_label(x_label, unit_x)
        if x_axis_item is not None:
            # setLabel with units turns SI prefixing back on, which is the whole
            # thing the custom axis exists to defeat.
            x_axis_item.enableAutoSIPrefix(False)
        self.set_y_label(y_label, unit_y)
        self.set_x_range(*x_range)
        self.set_y_range(*y_range)
        self.disable_context_menu()
        self.set_axis_limits(x_min, x_max, y_min, y_max)

        # Imposta font e colori direttamente qui
        tick_font = QFont("Arial", 14)
        label_font = QFont("Arial", 16, QFont.Bold)

        # Tick font (numeri assi)
        self.plot_widget.getAxis('bottom').setTickFont(tick_font)
        self.plot_widget.getAxis('left').setTickFont(tick_font)

        # Label font (nomi assi)
        self.plot_widget.getAxis('bottom').label.setFont(label_font)
        self.plot_widget.getAxis('left').label.setFont(label_font)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)







    # STILE
    def set_x_label(self, label, units): 
        self.plot_widget.setLabel('bottom', label, units)
    def set_y_label(self, label, units): 
        self.plot_widget.setLabel('left', label, units)

    # imposta la VISTA iniziale degli assi
    def set_x_range(self, min_val, max_val): 
        self.plot_widget.setXRange(min_val, max_val)
    def set_y_range(self, min_val, max_val):
        self.set_y_limits(y_min=min_val, y_max=max_val)
        self.plot_widget.setYRange(min_val, max_val, padding=0)

    def set_grid(self, show=True):
        self.plot_widget.showGrid(x=show, y=show, alpha=0.5)  # alpha è la trasparenza della griglia



    # 12. Impostazione dei limiti degli assi per zoom e pan
    def set_axis_limits(self, x_min=None, x_max=None, y_min=None, y_max=None):
        vb = self.plot_widget.getViewBox()
        limits = {}
        if x_min is not None:
            limits['xMin'] = x_min
        if x_max is not None:
            limits['xMax'] = x_max
        if y_min is not None:
            limits['yMin'] = y_min
        if y_max is not None:
            limits['yMax'] = y_max
        if limits:
            vb.setLimits(**limits)

    def set_y_limits(self, y_min=None, y_max=None):
        """
        Imposta i limiti dell'asse Y.
        Se y_min o y_max sono None, non vengono modificati.
        """
        vb = self.plot_widget.getViewBox()
        limits = {}
        if y_min is not None:
            limits['yMin'] = y_min
        if y_max is not None:
            limits['yMax'] = y_max
        if limits:
            vb.setLimits(**limits)

    def set_x_limits(self, x_min=None, x_max=None):
        """
        Imposta i limiti dell'asse X.
        Se x_min o x_max sono None, non vengono modificati.
        """
        vb = self.plot_widget.getViewBox()
        limits = {}
        if x_min is not None:
            limits['xMin'] = x_min
        if x_max is not None:
            limits['xMax'] = x_max
        if limits:
            vb.setLimits(**limits)


    # DATI (da sovrascrivere?)
    def add_curve(self, x=None, y=None, **kwargs):
        """
        Aggiunge o aggiorna la curva principale.
        Puoi passare tutti i parametri supportati da pyqtgraph.plot().
        Esempio: add_curve(x=x, y=y, pen='r', symbol='o', name='Segnale')
        """
        if x is not None and y is not None:
            self.plot.setData(x, y, **kwargs)
        else:
            self.plot.setData(**kwargs)
        #esemio di utilizzo:
        #self.add_curve(x=self.data_x, y=self.data_y, pen='r', symbol='o', name='Segnale')


    def clear_plot(self):
        self.plot.setData([], [])
        print("Grafico cancellato.")



    # IMPOSTAZIONI GENERALI
    def disable_context_menu(self): 
        self.plot_widget.setMenuEnabled(False)





    # AGGIUNGI NUOVA CURVA
    def add_threshold(self, x=None, y=None, name="Threshold", pen=pg.mkPen('red'), **kwargs):
        """
        Aggiunge una nuova curva (es soglia) al grafico.
        Puoi passare tutti i parametri supportati da pyqtgraph.plot().
        """
        return self.plot_widget.plot(x=x, y=y, pen=pen, name=name, **kwargs)
    


    #RIMOZIONE CURVA
    def remove_curve(self, curve):
        """Rimuove una curva dal grafico."""
        self.plot_widget.removeItem(curve)


    # FUNZIONE SPECIFICA PER GRAFICI CON DOMINIO TEMPORALE
    def update_time_window(self, x_data_seconds, xmin=0, window_size=16):
        """
        Mantiene la vista del grafico centrata sugli ultimi window_size secondi.
        x_data: array-like, i valori temporali (asse X)
        window_size: larghezza della finestra in secondi
        """
        # Filtra i NaN
        valid_x = x_data_seconds[~np.isnan(x_data_seconds)]
        if len(valid_x) == 0:
            print("Nessun valore valido per aggiornare la finestra temporale.")
            return
        if valid_x[-1] < window_size:
            xmin = xmin
            xmax = window_size
            self.set_x_range(xmin, xmax)
            return
        xmax = valid_x[-1]
        xmin = max(valid_x[0], xmax - window_size)
        self.set_x_range(xmin, xmax)
        