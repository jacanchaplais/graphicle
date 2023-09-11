graphicle
=========

|PyPI version| |Tests| |Documentation| |License| |pre-commit| |Code style:
black|

Utilities for representing high energy physics data as graphs /
networks.

Installation
------------

.. code:: bash

   pip install graphicle

Features
========

Object oriented interface to track-level particle data for collider
physics, with routines for constructing and performing calculations over
graph-structured data.

Provides data structures for:

* 4-momenta
* PDG codes
* Particle status codes
* Color codes
* Helicity / spin polarisation data
* COO adjacency lists (for graph-structured data)

.. code:: python3

   >>> import graphicle as gcl

   # query pdg records
   >>> pdgs = gcl.PdgArray([1, 3, 6, -6, 25, 2212])
   >>> pdgs.name
   ['d', 's', 't', 't~', 'H0', 'p'], dtype=object)
   >>> pdgs.charge
   array([-0.33333333, -0.33333333,  0.66666667, -0.66666667,  0.        ,
           1.        ])

   # extract information from momentum data
   >>> pmu_data
   array([( 1.95057378e-02,  3.12923088e-02,  3.53556064e-01, 3.55473730e-01),
          ( 2.60116947e+01, -3.63466398e+00, -3.33718718e+00, 2.64755711e+01),
          ( 5.91884324e-05, -7.62144267e-06, -6.76385314e-06, 6.00591927e-05),
          ( 2.82881807e+01,  4.32224823e+00,  2.14691072e+02, 2.16589841e+02),
          (-8.73280642e-02, -6.48540201e-02,  3.73744945e-01, 6.28679140e-01),
          ( 1.06204871e-01,  5.78888984e-01, -1.44899819e+02, 1.44901081e+02)],
         dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('e', '<f8')])
   >>> pmu = gcl.MomentumArray(pmu_data)
   ... pmu
   MomentumArray([[ 1.95057378e-02  3.12923088e-02  3.53556064e-01  3.55473730e-01]
                  [ 2.60116947e+01 -3.63466398e+00 -3.33718718e+00  2.64755711e+01]
                  [ 5.91884324e-05 -7.62144267e-06 -6.76385314e-06  6.00591927e-05]
                  [ 2.82881807e+01  4.32224823e+00  2.14691072e+02  2.16589841e+02]
                  [-8.73280642e-02 -6.48540201e-02  3.73744945e-01  6.28679140e-01]
                  [ 1.06204871e-01  5.78888984e-01 -1.44899819e+02  1.44901081e+02]],
                 dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('e', '<f8')])
   >>> pmu.pt
   array([3.68738715e-02, 2.62644064e+01, 5.96771055e-05, 2.86164812e+01,
          1.08776076e-01, 5.88550704e-01])
   >>> pmu.mass
   array([-7.45058060e-09,  5.11000489e-04,  9.09494702e-13,  5.10991478e-04,
           4.93680000e-01,  1.39570000e-01])
   >>> pmu.eta
   array([ 2.95639434, -0.12672178, -0.11309956,  2.71277683,  1.94796328,
          -6.1992861 ])
   >>> pmu.phi
   array([ 1.01339184, -0.138833  , -0.12806107,  0.15162078, -2.5028134 ,
           1.38935084])

   # calculate the inter-particle distances
   >>> pmu.delta_R(pmu)
   array([[0.        , 3.2913868 , 3.27485993, 0.89554388, 2.94501476,
           9.16339617],
          [3.2913868 , 0.        , 0.01736661, 2.85431528, 3.14526968,
           6.26189934],
          [3.27485993, 0.01736661, 0.        , 2.83968296, 3.14442819,
           6.27249595],
          [0.89554388, 2.85431528, 2.83968296, 0.        , 2.76241933,
           8.99760198],
          [2.94501476, 3.14526968, 3.14442819, 2.76241933, 0.        ,
           8.4908571 ],
          [9.16339617, 6.26189934, 6.27249595, 8.99760198, 8.4908571 ,
           0.        ]])

Graphicle really shines with its composite data structures. These can be
used to filter and query heterogeneous particle data records
simultaneously, either using user provided boolean masks, or
``MaskArray`` instances produced with routines in the ``select`` module.
Additionally, routines in the ``calculate`` and ``transform`` modules
take composite data structures to standardise useful calculations which
blends multiple particle data records.

To see an example, let’s generate a collision event using Pythia,
wrapped with ``showerpipe``.

.. code:: python3

   >>> from showerpipe.generator import PythiaGenerator
   ...
   ... lhe_path = "https://zenodo.org/record/6034610/files/unweighted_events.lhe.gz"
   ... gen = PythiaGenerator("pythia-settings.cmnd", lhe_path, 1)
   >>> for event in gen:
   ...     graph = gcl.Graphicle.from_event(event)
   ...     break

   >>> print(graph)
   name            px          py          pz      energy    color    anticolor    helicity    status  final      src    dst
   p         0.00E+00    0.00E+00    6.50E+03    6.50E+03        0            0           9       -12  False        0     -1
   p         0.00E+00    0.00E+00   -6.50E+03    6.50E+03        0            0           9       -12  False        0     -2
   g         0.00E+00    0.00E+00    2.99E+02    2.99E+02      503          502           1       -21  False       -6     -3
   g        -0.00E+00   -0.00E+00   -5.99E+02    5.99E+02      501          503           1       -21  False       -7     -3
   t         2.34E+02   -2.20E+01   -4.76E+02    5.58E+02      501            0           0       -22  False       -3     -4
   ...     ...         ...         ...         ...             ...          ...         ...       ...  ...        ...    ...
   gamma     1.30E-02   -1.30E+00   -3.24E+00    3.49E+00        0            0           9        91  True      -969    979
   gamma     1.70E-01   -8.21E-01   -2.32E+00    2.47E+00        0            0           9        91  True      -970    980
   gamma     3.12E-01   -2.26E+00   -6.82E+00    7.19E+00        0            0           9        91  True      -970    981
   gamma     9.38E-03   -3.58E-01   -7.98E-01    8.75E-01        0            0           9        91  True      -971    982
   gamma     3.08E-02   -4.36E-02   -4.56E-02    7.02E-02        0            0           9        91  True      -971    983

   [1065 particles × 12 attributes]
   >>> graph.pdg
   PdgArray([2212 2212   21 ...   22   22   22], dtype=int32)
   >>> graph.adj
   AdjacencyList([[   0   -1]
                  [   0   -2]
                  [  -6   -3]
                  ...
                  [-970  981]
                  [-971  982]
                  [-971  983]],
                 dtype=[('src', '<i4'), ('dst', '<i4')])

   # select all descendants of the W bosons from the hard process
   >>> W_mask = gcl.select.hard_descendants(graph, {24})
   >>> W_mask
   MaskGroup(mask_arrays=["W+", "W-"], agg_op=OR)
   # filter data record to get final state W+ boson descendants
   >>> Wp_desc = graph[W_mask["W+"] & graph.final]
   >>> print(Wp_desc)
   name            px         py         pz    energy    color    anticolor    helicity    status  final      src    dst
   gamma     2.46E-05  -5.65E-06  -1.54E-05  2.95E-05        0            0           9        51  True      -350    353
   nu(tau)   1.72E+02   3.52E+01  -3.18E+02  3.63E+02        0            0           9        52  True      -351    354
   nu(tau)~  1.73E+01  -4.48E+00  -1.08E+01  2.09E+01        0            0           9        91  True      -352    687
   pi+       1.19E+01  -3.15E+00  -7.51E+00  1.44E+01        0            0           9        91  True      -352    690
   gamma     4.12E+00  -1.09E+00  -2.19E+00  4.79E+00        0            0           9        91  True      -688    879
   gamma     1.54E+00  -4.72E-01  -8.87E-01  1.84E+00        0            0           9        91  True      -688    880
   gamma     2.11E+00  -4.94E-01  -9.96E-01  2.38E+00        0            0           9        91  True      -689    881
   gamma     3.22E+00  -7.42E-01  -1.71E+00  3.72E+00        0            0           9        91  True      -689    882

   [8 particles × 12 attributes]

   # numpy can interface with graphicle - let's sum the momenta
   >>> Wp_sum = np.sum(Wp_desc.pmu, axis=0)
   >>> Wp_sum.mass
   80.419002446

More information on the API is available in the
`documentation <https://graphicle.readthedocs.io>`__

.. |PyPI version| image:: https://img.shields.io/pypi/v/graphicle.svg
   :target: https://pypi.org/project/graphicle/
.. |Tests| image:: https://github.com/jacanchaplais/graphicle/actions/workflows/tests.yml/badge.svg
.. |Documentation| image:: https://readthedocs.org/projects/graphicle/badge/?version=latest
   :target: https://graphicle.readthedocs.io
.. |License| image:: https://img.shields.io/pypi/l/graphicle
   :target: https://raw.githubusercontent.com/jacanchaplais/graphicle/main/LICENSE.txt
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
   :target: https://github.com/pre-commit/pre-commit
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
