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
Mixin per componenti PlantLeaf che NON seguono il tema globale.
Qui puoi implementare logica di stile, palette o comportamento custom.
    """
from PySide6.QtWidgets import QSplitter


class SpecialComponent:
    def set_custom_style(self):
        """Override nei figli per applicare uno stile personalizzato"""
        pass

    def set_custom_palette(self, palette):
        """Applica una palette personalizzata"""
        self.setPalette(palette)

    def set_custom_font(self, font):
        """Applica un font personalizzato"""
        self.setFont(font)


    # Puoi aggiungere altri metodi per gestire eventi, animazioni, ecc.



#FUROI DALLA CLASSE

# METODO UNIVERSALE PER SOSTITUIRE COMPONENTE DEL .. .ui CON COMPONENTE PERSONALIZZATO
# utils/widget_utils.py
def replace_widget(parent, old_widget_name: str, new_widget):
    old_widget = getattr(parent, old_widget_name, None)
    if old_widget is None:
        print(f"❌ Widget '{old_widget_name}' non trovato!")
        return

    parent_widget = old_widget.parentWidget()
    if isinstance(parent_widget, QSplitter):
        index = parent_widget.indexOf(old_widget)
        parent_widget.insertWidget(index, new_widget)
        old_widget.setParent(None)
        setattr(parent, old_widget_name, new_widget)
        print(f"✅ Sostituito '{old_widget_name}' con il nuovo widget custom (QSplitter).")
        return

    parent_layout = parent_widget.layout()
    if parent_layout is None:
        print(f"❌ Layout padre non trovato per '{old_widget_name}'!")
        return
    for i in range(parent_layout.count()):
        if parent_layout.itemAt(i).widget() == old_widget:
            parent_layout.removeWidget(old_widget)
            old_widget.setParent(None)
            parent_layout.insertWidget(i, new_widget)
            setattr(parent, old_widget_name, new_widget)
            print(f"✅ Sostituito '{old_widget_name}' con il nuovo widget custom.")
            return
    print(f"❌ Widget '{old_widget_name}' non trovato nel layout!")