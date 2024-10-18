# hsg_data_validation: Funktionen für die Datenvalidierung
# Copyright (C) 2024 HSGSim Arbeitsgruppe "Messdaten und Machine Learning"
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt


def plot_scatter(x, y, figure_size=(12, 7)):
    plt.rcParams["figure.figsize"] = figure_size
    plt.scatter(x=x, y=y, alpha=0.6, s=10)
    plt.show()


def _draw_as_table(df, pagesize):
    alternating_colors = [['white'] * len(df.columns), ['lightgray'] * len(df.columns)] * len(df)
    alternating_colors = alternating_colors[:len(df)]
    fig, ax = plt.subplots(figsize=pagesize)
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=df.values,
                         rowLabels=df.index,
                         colLabels=df.columns,
                         rowColours=['lightblue'] * len(df),
                         colColours=['lightblue'] * len(df.columns),
                         cellColours=alternating_colors,
                         loc='center')
    return fig
