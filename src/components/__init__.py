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

"""
Components module - Contains the main components of the PlantLeaf application.
"""

from .start_stop_button import StartStopButton
from .data_table import DataTable
from .sampling_settings import VoltageSamplingSettingsPopup, AudioSamplingSettingsPopup
from .choose_serial_port import ChooseSerialPort  
from .time_input_widget import TimeInputWidget


__all__ = [
    'StartStopButton',
    'DataTable',
    'VoltageSamplingSettingsPopup',
    'AudioSamplingSettingsPopup',
    'ChooseSerialPort',
    'TimeInputWidget',
]