"""Functions with nilearn functionality customized
(adpated from https://github.com/nilearn/nilearn)
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrow
from matplotlib.lines import Line2D
from scipy.sparse import issparse
from scipy.stats import scoreatpercentile
from nilearn.plotting import cm, plot_glass_brain
from nilearn._utils.param_validation import check_threshold
from nilearn.plotting.displays._axes import _coords_3d_to_2d

def plot_connectome(adjacency_matrix, node_coords,
                    node_color='auto', node_size=50,
                    edge_cmap=cm.bwr,
                    edge_vmin=None, edge_vmax=None,
                    edge_threshold=None, edge_norm=None,
                    output_file=None, display_mode='ortho',
                    figure=None, axes=None, title=None,
                    annotate=True, black_bg=False,
                    alpha=0.7,
                    edge_kwargs=None, node_kwargs=None,
                    colorbar=False):
    """Plot connectome on top of the brain glass schematics.

    The plotted image should be in MNI space for this function to work
    properly.

    In the case of 'l' and 'r' directions (for hemispheric projections),
    markers in the coordinate x == 0 are included in both hemispheres.

    Parameters
    ----------
    adjacency_matrix : numpy array of shape (n, n)
        Represents the link strengths of the graph. The matrix can be
        symmetric which will result in an undirected graph, or not
        symmetric which will result in a directed graph.

    node_coords : numpy array_like of shape (n, 3)
        3d coordinates of the graph nodes in world space.

    node_color : color or sequence of colors or 'auto', optional
        Color(s) of the nodes. If string is given, all nodes
        are plotted with same color given in string.

    node_size : scalar or array_like, optional
        Size(s) of the nodes in points^2. Default=50.

    edge_cmap : colormap, optional
        Colormap used for representing the strength of the edges.
        Default=cm.bwr.

    edge_vmin, edge_vmax : float, optional
        If not None, either or both of these values will be used to
        as the minimum and maximum values to color edges. If None are
        supplied the maximum absolute value within the given threshold
        will be used as minimum (multiplied by -1) and maximum
        coloring levels.

    edge_threshold : str or number, optional
        If it is a number only the edges with a value greater than
        edge_threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%%", and only the edges with a abs(value) above
        the given percentile will be shown.
    
    edge_norm : :class:`~matplotlib.colors.Normalize`, optional
        The normalization method to use before mapping colors based on
        the colormap.
    %(output_file)s
    display_mode : string, optional
        Choose the direction of the cuts: 'x' - sagittal, 'y' - coronal,
        'z' - axial, 'l' - sagittal left hemisphere only,
        'r' - sagittal right hemisphere only, 'ortho' - three cuts are
        performed in orthogonal directions. Possible values are: 'ortho',
        'x', 'y', 'z', 'xz', 'yx', 'yz', 'l', 'r', 'lr', 'lzr', 'lyr',
        'lzry', 'lyrz'. Default='ortho'.
    %(figure)s
    %(axes)s
    %(title)s
    %(annotate)s
    %(black_bg)s
        Default=False.
    alpha : float between 0 and 1, optional
        Alpha transparency for the brain schematics. Default=0.7.

    edge_kwargs : dict, optional
        Will be passed as kwargs for each edge matlotlib Line2D.

    node_kwargs : dict, optional
        Will be passed as kwargs to the plt.scatter call that plots all
        the nodes in one go.
    %(colorbar)s
        Default=False.

    See Also
    --------
    nilearn.plotting.find_parcellation_cut_coords : Extraction of node
        coords on brain parcellations.
    nilearn.plotting.find_probabilistic_atlas_cut_coords : Extraction of
        node coords on brain probabilistic atlases.

    """
    display = plot_glass_brain(None,
                               display_mode=display_mode,
                               figure=figure, axes=axes, title=title,
                               annotate=annotate,
                               black_bg=black_bg,
                               alpha=alpha)

    add_graph(display, adjacency_matrix, node_coords,
              node_color=node_color, node_size=node_size,
              edge_cmap=edge_cmap,
              edge_vmin=edge_vmin, edge_vmax=edge_vmax,
              edge_threshold=edge_threshold, edge_norm=edge_norm,
              edge_kwargs=edge_kwargs, node_kwargs=node_kwargs,
              colorbar=colorbar)

    if output_file is not None:
        display.savefig(output_file)
        display.close()
        display = None

    return display

def add_graph(display_obj, adjacency_matrix, node_coords,
              node_color='auto', node_size=50,
              edge_cmap=cm.bwr,
              edge_vmin=None, edge_vmax=None,
              edge_threshold=None, edge_norm=None,
              edge_kwargs=None, node_kwargs=None, colorbar=False,
              ):
    """Plot undirected graph on each of the axes.

    Parameters
    ----------
    adjacency_matrix : :class:`numpy.ndarray` of shape ``(n, n)``
        Represents the edges strengths of the graph.
        The matrix can be symmetric which will result in
        an undirected graph, or not symmetric which will
        result in a directed graph.

    node_coords : :class:`numpy.ndarray` of shape ``(n, 3)``
        3D coordinates of the graph nodes in world space.

    node_color : color or sequence of colors, optional
        Color(s) of the nodes. Default='auto'.

    node_size : scalar or array_like, optional
        Size(s) of the nodes in points^2. Default=50.

    edge_cmap : :class:`~matplotlib.colors.Colormap`, optional
        Colormap used for representing the strength of the edges.
        Default=cm.bwr.

    edge_vmin, edge_vmax : :obj:`float`, optional
        - If not ``None``, either or both of these values will be used
            to as the minimum and maximum values to color edges.
        - If ``None`` are supplied, the maximum absolute value within the
            given threshold will be used as minimum (multiplied by -1) and
            maximum coloring levels.

    edge_threshold : :obj:`str` or :obj:`int` or :obj:`float`, optional
        - If it is a number only the edges with a value greater than
            ``edge_threshold`` will be shown.
        - If it is a string it must finish with a percent sign,
            e.g. "25.3%", and only the edges with a abs(value) above
            the given percentile will be shown.

    edge_norm : :class:`~matplotlib.colors.Normalize`, optional
        The normalization method to use before mapping colors based on
        the colormap.

    edge_kwargs : :obj:`dict`, optional
        Will be passed as kwargs for each edge
        :class:`~matplotlib.lines.Line2D`.

    node_kwargs : :obj:`dict`
        Will be passed as kwargs to the function
        :func:`~matplotlib.pyplot.scatter` which plots all the
        nodes at one.
    """
    # set defaults
    if edge_kwargs is None:
        edge_kwargs = {}
    if node_kwargs is None:
        node_kwargs = {}
    if isinstance(node_color, str) and node_color == 'auto':
        nb_nodes = len(node_coords)
        node_color = mpl_cm.Set2(np.linspace(0, 1, nb_nodes))
    node_coords = np.asarray(node_coords)

    # decompress input matrix if sparse
    if issparse(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.toarray()

    # make the lines below well-behaved
    adjacency_matrix = np.nan_to_num(adjacency_matrix)

    # safety checks
    if 's' in node_kwargs:
        raise ValueError("Please use 'node_size' and not 'node_kwargs' "
                            "to specify node sizes")
    if 'c' in node_kwargs:
        raise ValueError("Please use 'node_color' and not 'node_kwargs' "
                            "to specify node colors")

    adjacency_matrix_shape = adjacency_matrix.shape
    if (len(adjacency_matrix_shape) != 2
            or adjacency_matrix_shape[0] != adjacency_matrix_shape[1]):
        raise ValueError(
            "'adjacency_matrix' is supposed to have shape (n, n)."
            ' Its shape was {0}'.format(adjacency_matrix_shape))

    node_coords_shape = node_coords.shape
    if len(node_coords_shape) != 2 or node_coords_shape[1] != 3:
        message = (
            "Invalid shape for 'node_coords'. You passed an "
            "'adjacency_matrix' of shape {0} therefore "
            "'node_coords' should be a array with shape ({0[0]}, 3) "
            'while its shape was {1}').format(adjacency_matrix_shape,
                                                node_coords_shape)

        raise ValueError(message)

    if isinstance(node_color, (list, np.ndarray)) and len(node_color) != 1:
        if len(node_color) != node_coords_shape[0]:
            raise ValueError(
                "Mismatch between the number of nodes ({0}) "
                "and the number of node colors ({1})."
                .format(node_coords_shape[0], len(node_color)))

    if node_coords_shape[0] != adjacency_matrix_shape[0]:
        raise ValueError(
            "Shape mismatch between 'adjacency_matrix' "
            "and 'node_coords'"
            "'adjacency_matrix' shape is {0}, 'node_coords' shape is {1}"
            .format(adjacency_matrix_shape, node_coords_shape))

    # If the adjacency matrix is not symmetric, give a warning
    symmetric = True
    if not np.allclose(adjacency_matrix, adjacency_matrix.T, rtol=1e-3):
        symmetric = False
        warnings.warn(("'adjacency_matrix' is not symmetric. "
                        "A directed graph will be plotted."))

    # For a masked array, masked values are replaced with zeros
    if hasattr(adjacency_matrix, 'mask'):
        if not (adjacency_matrix.mask == adjacency_matrix.mask.T).all():
            symmetric = False
            warnings.warn(("'adjacency_matrix' was masked with "
                            "a non symmetric mask. A directed "
                            "graph will be plotted."))
        adjacency_matrix = adjacency_matrix.filled(0)

    if edge_threshold is not None:
        if symmetric:
            # Keep a percentile of edges with the highest absolute
            # values, so only need to look at the covariance
            # coefficients below the diagonal
            lower_diagonal_indices = np.tril_indices_from(adjacency_matrix,
                                                            k=-1)
            lower_diagonal_values = adjacency_matrix[
                lower_diagonal_indices]
            edge_threshold = check_threshold(
                edge_threshold, np.abs(lower_diagonal_values),
                scoreatpercentile, 'edge_threshold')
        else:
            edge_threshold = check_threshold(
                edge_threshold, np.abs(adjacency_matrix.ravel()),
                scoreatpercentile, 'edge_threshold')

        adjacency_matrix = adjacency_matrix.copy()
        threshold_mask = np.abs(adjacency_matrix) < edge_threshold
        adjacency_matrix[threshold_mask] = 0

    if symmetric:
        lower_triangular_adjacency_matrix = np.tril(adjacency_matrix, k=-1)
        non_zero_indices = lower_triangular_adjacency_matrix.nonzero()
    else:
        non_zero_indices = adjacency_matrix.nonzero()

    line_coords = [node_coords[list(index)]
                    for index in zip(*non_zero_indices)]

    adjacency_matrix_values = adjacency_matrix[non_zero_indices]
    for ax in display_obj.axes.values():
        ax._add_markers(node_coords, node_color, node_size, **node_kwargs)
        if line_coords:
            _add_lines(ax, line_coords, adjacency_matrix_values, edge_cmap,
                            norm=edge_norm, vmin=edge_vmin, vmax=edge_vmax,
                            directed=(not symmetric),
                            **edge_kwargs)
        # To obtain the brain left view, we simply invert the x axis
        if(ax.direction == 'l'
            and not (ax.ax.get_xlim()[0] > ax.ax.get_xlim()[1])):
            ax.ax.invert_xaxis()

    if colorbar:
        display_obj._colorbar = colorbar
        display_obj._show_colorbar(ax.cmap, ax.norm, threshold=edge_threshold)

    plt.draw_if_interactive()

    return None

def _add_lines(ax_obj, line_coords, line_values, cmap, norm=None,
                vmin=None, vmax=None, directed=False, **kwargs):
    """Plot lines

    Parameters
    ----------
    line_coords : :obj:`list` of :class:`numpy.ndarray` of shape (2, 3)
        3D coordinates of lines start points and end points.

    line_values : array_like
        Values of the lines.

    cmap : :class:`~matplotlib.colors.Colormap`
        Colormap used to map ``line_values`` to a color.

    norm : :class:`~matplotlib.colors.Normalize`, optional
        The normalization method to use before mapping colors based on
        the colormap.

    vmin, vmax : :obj:`float`, optional
        If not ``None``, either or both of these values will be used to
        as the minimum and maximum values to color lines. If ``None`` are
        supplied the maximum absolute value within the given threshold
        will be used as minimum (multiplied by -1) and maximum
        coloring levels.

    directed : :obj:`bool`, optional
        Add arrows instead of lines if set to ``True``.
        Use this when plotting directed graphs for example.
        Default=False.

    kwargs : :obj:`dict`
        Additional arguments to pass to :class:`~matplotlib.lines.Line2D`.

    """
    # colormap for colorbar
    ax_obj.cmap = cmap
    if vmin is None and vmax is None:
        abs_line_values_max = np.abs(line_values).max()
        vmin = -abs_line_values_max
        vmax = abs_line_values_max
    elif vmin is None:
        if vmax > 0:
            vmin = -vmax
        else:
            raise ValueError(
                "If vmax is set to a non-positive number "
                "then vmin needs to be specified"
            )
    elif vmax is None:
        if vmin < 0:
            vmax = -vmin
        else:
            raise ValueError(
                "If vmin is set to a non-negative number "
                "then vmax needs to be specified"
            )
    if norm is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
        # normalization useful for colorbar
    ax_obj.norm = norm
    abs_norm = Normalize(vmin=0, vmax=max(abs(vmax), abs(vmin)))
    value_to_color = plt.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba

    # Allow lines only in their respective hemisphere when appropriate
    if ax_obj.direction in 'lr':
        relevant_lines = []
        for lidx, line in enumerate(line_coords):
            if ax_obj.direction == 'r':
                if line[0, 0] >= 0 and line[1, 0] >= 0:
                    relevant_lines.append(lidx)
            elif ax_obj.direction == 'l':
                if line[0, 0] < 0 and line[1, 0] < 0:
                    relevant_lines.append(lidx)
        line_coords = np.array(line_coords)[relevant_lines]
        line_values = line_values[relevant_lines]

    for start_end_point_3d, line_value in zip(
            line_coords, line_values):
        start_end_point_2d = _coords_3d_to_2d(start_end_point_3d,
                                                ax_obj.direction)

        color = value_to_color(line_value)
        abs_line_value = abs(line_value)
        linewidth = 1 + 2 * abs_norm(abs_line_value)
        # Hacky way to put the strongest connections on top of the weakest
        # note sign does not matter hence using 'abs'
        zorder = 10 + 10 * abs_norm(abs_line_value)
        this_kwargs = {'color': color, 'linewidth': linewidth,
                        'zorder': zorder}
        # kwargs should have priority over this_kwargs so that the
        # user can override the default logic
        this_kwargs.update(kwargs)
        xdata, ydata = start_end_point_2d.T
        # If directed is True, add an arrow
        if directed:
            dx = xdata[1] - xdata[0]
            dy = ydata[1] - ydata[0]
            # Hack to avoid empty arrows to crash with
            # matplotlib versions older than 3.1
            # This can be removed once support for
            # matplotlib pre 3.1 has been dropped.
            if dx == dy == 0:
                arrow = FancyArrow(xdata[0], ydata[0],
                                    dx, dy)
            else:
                arrow = FancyArrow(xdata[0], ydata[0],
                                    dx, dy,
                                    length_includes_head=True,
                                    width=linewidth,
                                    head_width=3 * linewidth,
                                    **this_kwargs)
            ax_obj.ax.add_patch(arrow)
        # Otherwise a line
        else:
            line = Line2D(xdata, ydata, **this_kwargs)
            ax_obj.ax.add_line(line)

    return None