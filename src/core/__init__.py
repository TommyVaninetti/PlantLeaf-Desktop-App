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
Core module - Contiene la logica di base dell'applicazione PlantLeaf
"""

from .base_window import BaseWindow
from .font_manager import FontManager
from .theme_manager import ThemeManager
from .layout_manager import LayoutManager
from .settings_manager import SettingsManager
from .replay_base_window import ReplayBaseWindow
from .file_handler_mixin import FileHandlerMixin

__all__ = [
    'BaseWindow',
    'FontManager', 
    'ThemeManager',
    'LayoutManager',
    'SettingsManager',
    'ReplayBaseWindow',
    'FileHandlerMixin'
]